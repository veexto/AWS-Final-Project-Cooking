"""
Bep Thong Minh - AWS Lambda API (Teaching & Cooking Assistant)
Runtime: Python 3.12
Handler: lambda_function.lambda_handler
DynamoDB table: RecipesTable (partition key RecipeID, String)
Integrations: Amazon DynamoDB, Amazon Location Service (geo-places v2), Amazon Bedrock
Supports API Gateway REST API v1 and HTTP API v2 events.
"""
import base64
import json
import logging
import math
import os
import random
import unicodedata
import urllib.error
import urllib.request
from decimal import Decimal
from urllib.parse import parse_qs, quote

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", REGION).strip()
TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "RecipesTable").strip()
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "apac.amazon.nova-micro-v1:0").strip()
BEDROCK_API_KEY = os.environ.get("BEDROCK_API_KEY", "").strip()

MAX_INPUT_LENGTH = 200
MAX_KEYWORDS = 10

dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(TABLE_NAME)
geo_places_client = boto3.client("geo-places", region_name=REGION)
bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Content-Type": "application/json; charset=utf-8",
}


def json_default(value):
    if isinstance(value, Decimal):
        return int(value) if value % 1 == 0 else float(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps(body, ensure_ascii=False, default=json_default),
    }


def normalize(value):
    text = str(value or "").strip().lower()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def csv_values(value, field_name):
    if not value:
        return []
    if len(value) > MAX_INPUT_LENGTH:
        raise ValueError(f"Tham số {field_name} vượt quá {MAX_INPUT_LENGTH} ký tự.")
    values = [part.strip() for part in value.split(",") if part.strip()]
    if any(len(part) > 60 for part in values):
        raise ValueError(f"Mỗi giá trị trong {field_name} tối đa 60 ký tự.")
    return [v for v in values[:MAX_KEYWORDS] if not any(c in v for c in "<>{}$`\\;")]


def query_params(event):
    params = event.get("queryStringParameters") or {}
    if params:
        return params
    raw = event.get("rawQueryString", "")
    return {k: v[0] for k, v in parse_qs(raw).items() if v}


def parse_json_body(event):
    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    try:
        return json.loads(body)
    except (TypeError, json.JSONDecodeError):
        raise ValueError("Dữ liệu gửi lên không phải JSON hợp lệ.")


def get_all_recipes():
    items = []
    kwargs = {}
    while True:
        result = table.scan(**kwargs)
        items.extend(result.get("Items", []))
        last_key = result.get("LastEvaluatedKey")
        if not last_key:
            break
        kwargs = {"ExclusiveStartKey": last_key}
    items.sort(key=lambda x: str(x.get("RecipeID", "")))
    logger.info("Loaded %s recipes from %s", len(items), TABLE_NAME)
    return items


def has_any(values, search_terms):
    normalized_values = [normalize(v) for v in values]
    for term in search_terms:
        t = normalize(term)
        if any(t in v or v in t for v in normalized_values if v and t):
            return True
    return False


def has_tag(recipe, field, tag):
    return tag in (recipe.get(field) or [])


def matches(recipe, filters):
    ingredients = recipe.get("Ingredients") or []
    allergens = recipe.get("Allergens") or []
    nutrition = recipe.get("Nutrition") or {}

    if filters["ingredients"] and not has_any(ingredients, filters["ingredients"]):
        return False
    if filters["health"] and not has_tag(recipe, "HealthTags", filters["health"]):
        return False
    if filters["need"] and not (
        has_tag(recipe, "HealthTags", filters["need"])
        or has_tag(recipe, "DietTags", filters["need"])
    ):
        return False
    if filters["mood"] and not has_tag(recipe, "MoodTags", filters["mood"]):
        return False
    if filters["allergies"] and (
        has_any(ingredients, filters["allergies"])
        or has_any(allergens, filters["allergies"])
    ):
        return False
    if filters["avoid"] and has_any(ingredients, filters["avoid"]):
        return False

    numeric_rules = (
        ("maxTime", "CookingTimeMinutes"),
        ("maxBudget", "EstimatedCostVND"),
    )
    for param, field in numeric_rules:
        if filters[param] is not None:
            try:
                value = float(recipe.get(field))
            except (TypeError, ValueError):
                return False
            if value > filters[param]:
                return False

    for param, field in (("maxSodium", "SodiumMg"), ("maxCarbs", "CarbsG")):
        if filters[param] is not None:
            try:
                value = float(nutrition[field])
            except (KeyError, TypeError, ValueError):
                return False
            if value > filters[param]:
                return False
    return True


def ingredient_score(recipe, keywords):
    ingredients = [normalize(v) for v in (recipe.get("Ingredients") or [])]
    return sum(1 for term in keywords if any(normalize(term) in ing for ing in ingredients))


def haversine_meters(lat1, lon1, lat2, lon2):
    r = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return int(2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


# 1. TÌM CỬA HÀNG THỰC PHẨM GẦN VỊ TRÍ (AMAZON LOCATION GEO-PLACES V2)
def handle_nearby_stores(params):
    ingredients = (params.get("ingredients") or "thực phẩm tươi").strip()
    if len(ingredients) > MAX_INPUT_LENGTH:
        raise ValueError(f"Tham số ingredients vượt quá {MAX_INPUT_LENGTH} ký tự.")
    tier = (params.get("tier") or "mid").strip().lower()

    address = (params.get("address") or "").strip()
    if len(address) > 200:
        raise ValueError("Địa chỉ tìm kiếm tối đa 200 ký tự.")

    if address:
        bias_lng = float(params.get("lng", 106.7009))
        bias_lat = float(params.get("lat", 10.7769))
        geocode = geo_places_client.search_text(
            QueryText=f"{address}, Việt Nam",
            BiasPosition=[bias_lng, bias_lat],
            MaxResults=1,
            Language="vi",
            Filter={
                "BoundingBox": [102.0, 8.0, 110.0, 24.0],
                "IncludeCountries": ["VNM"]
            }
        )
        address_results = geocode.get("ResultItems", [])
        if not address_results or not address_results[0].get("Position"):
            return response(200, {
                "source": "amazon-location-geo-places",
                "advice": "Không tìm thấy địa chỉ này. Hãy thử nhập địa chỉ cụ thể hơn hoặc dùng GPS.",
                "stores": [],
            })
        address_position = address_results[0]["Position"]
        lng, lat = float(address_position[0]), float(address_position[1])
    else:
        try:
            lat = float(params.get("lat", 10.7769))
            lng = float(params.get("lng", 106.7009))
        except (TypeError, ValueError):
            lat, lng = 10.7769, 106.7009

    tier_config = {
        "low": "Ngân sách tiết kiệm (< 35.000đ): Ưu tiên chợ truyền thống và siêu thị bình dân gần bạn.",
        "mid": "Ngân sách vừa phải (35.000đ – 70.000đ): Ưu tiên chợ, siêu thị và cửa hàng thực phẩm tổng hợp.",
        "high": "Ngân sách thoải mái (> 70.000đ): Ưu tiên siêu thị và cửa hàng thực phẩm tổng hợp có nhiều lựa chọn.",
    }
    advice = tier_config.get(tier, tier_config["mid"])

    excluded_name_terms = (
        "famima", "familymart", "family mart", "circle k", "gs25",
        "7-eleven", "ministop", "cheers", "shop&go", "shop and go", "fumart",
        "trai cay", "rau cu", "cua hang rau", "cua hang thit",
        "hai san", "ca tuoi", "me va be", "baby", "kids",
        "chao dinh duong", "mon ngon", "nha hang", "quan an",
        "dien may", "cua hang gao", "dai ly gao", "vua gao",
        "luong thuc", "rau sach", "vegetables", "fruits",
        "fruit garden", "nong san", "vinamilk", "nonglam milk",
        "ba huan", "trung sach", "farmbox",
    )
    excluded_category_terms = (
        "restaurant", "cafe", "coffee", "fast food", "food court",
        "bar", "pub", "gas station", "pet store", "clothing",
        "hotel", "lodging", "shopping mall", "shopping center",
    )
    grocery_name_terms = (
        "bach hoa xanh", "winmart", "win mart", "coop food", "co.op food",
        "coopmart", "co.opmart", "kingfood", "tops market", "lotte mart",
        "mega market", "go market", "emart", "satrafood",
        "food mart", "food store", "cua hang thuc pham",
        "tap hoa", "bach hoa", "sieu thi", "cho ",
    )

    try:
        stores_by_key = {}
        next_token = None
        page_count = 0
        max_pages = 5

        while page_count < max_pages:
            request_payload = {
                "QueryPosition": [lng, lat],
                "QueryRadius": 10000,
                "Filter": {"IncludeCountries": ["VNM"]},
                "MaxResults": 100,
                "Language": "vi",
            }
            if next_token:
                request_payload["NextToken"] = next_token

            result = geo_places_client.search_nearby(**request_payload)
            page_count += 1

            for item in result.get("ResultItems", []):
                categories = item.get("Categories") or []
                category_names = [
                    normalize(category.get("LocalizedName") or category.get("Name") or "")
                    for category in categories
                ]
                title = (item.get("Title") or "").strip()
                logger.info(
                    "Nearby POI raw: title=%r place_type=%r categories=%s",
                    title, item.get("PlaceType"), category_names
                )

                if item.get("PlaceType") != "PointOfInterest":
                    continue

                if not title:
                    continue
                title_norm = normalize(title)
                if any(term in title_norm for term in excluded_name_terms):
                    logger.info("Nearby POI rejected by name filter: %r", title)
                    continue

                if any(
                    excluded in category
                    for category in category_names
                    for excluded in excluded_category_terms
                ):
                    logger.info("Nearby POI rejected by excluded category: %r categories=%s", title, category_names)
                    continue

                has_grocery_name = any(term in title_norm for term in grocery_name_terms)
                if not has_grocery_name:
                    logger.info("Nearby POI rejected: not recognized as general grocery: %r categories=%s", title, category_names)
                    continue

                pos = item.get("Position")
                if not pos or len(pos) < 2:
                    continue

                store_lat, store_lng = float(pos[1]), float(pos[0])
                dist_m = haversine_meters(lat, lng, store_lat, store_lng)
                if dist_m > 10000:
                    continue

                addr_label = (item.get("Address") or {}).get("Label") or ""
                if not addr_label:
                    continue

                chain_names = [
                    chain.get("Name", "")
                    for chain in (item.get("BusinessChains") or [])
                    if chain.get("Name")
                ]
                category_label = ", ".join(
                    category.get("LocalizedName") or category.get("Name", "")
                    for category in categories
                    if category.get("LocalizedName") or category.get("Name")
                )
                store = {
                    "name": title,
                    "address": addr_label,
                    "distanceMeters": dist_m,
                    "lat": store_lat,
                    "lng": store_lng,
                    "category": category_label,
                    "businessChains": chain_names,
                }

                address_key = normalize(addr_label)
                normalized_title = normalize(title)
                known_brands = (
                    "coopmart", "coop food", "winmart", "bach hoa xanh",
                    "kingfoodmart", "tops market", "lotte mart", "mega market",
                    "emart", "satrafood",
                )
                brand_key = next((brand for brand in known_brands if brand in normalized_title), None)
                place_id = item.get("PlaceId")
                key = (brand_key, address_key) if brand_key else (
                    place_id or (normalized_title, address_key)
                )
                stores_by_key[key] = store

            next_token = result.get("NextToken")
            if not next_token:
                break

        if next_token:
            logger.warning(
                "SearchNearby reached page limit (%s); more results may exist. location=(%s,%s)",
                max_pages, lat, lng
            )

        if not stores_by_key:
            fallback_queries = (
                "siêu thị", "chợ", "cửa hàng thực phẩm",
                "supermarket", "grocery store", "food market",
            )
            for search_term in fallback_queries:
                try:
                    text_result = geo_places_client.search_text(
                        QueryText=search_term,
                        MaxResults=100,
                        Filter={
                            "Circle": {"Center": [lng, lat], "Radius": 10000},
                            "IncludeCountries": ["VNM"],
                        },
                        Language="vi",
                    )
                    logger.info(
                        "SearchText fallback query=%r raw_results=%s",
                        search_term, len(text_result.get("ResultItems", []))
                    )

                    for item in text_result.get("ResultItems", []):
                        title = (item.get("Title") or "").strip()
                        if not title:
                            continue
                        title_norm = normalize(title)
                        if any(term in title_norm for term in excluded_name_terms):
                            continue

                        categories = item.get("Categories") or []
                        category_names = [
                            normalize(category.get("LocalizedName") or category.get("Name") or "")
                            for category in categories
                        ]
                        if any(
                            excluded in category
                            for category in category_names
                            for excluded in excluded_category_terms
                        ):
                            continue

                        has_grocery_name = any(
                            term in title_norm for term in grocery_name_terms
                        )
                        if not has_grocery_name:
                            continue

                        pos = item.get("Position")
                        if not pos or len(pos) < 2:
                            continue
                        store_lat, store_lng = float(pos[1]), float(pos[0])
                        dist_m = haversine_meters(lat, lng, store_lat, store_lng)
                        if dist_m > 10000:
                            continue

                        addr_label = (item.get("Address") or {}).get("Label") or ""
                        if not addr_label:
                            continue
                        chain_names = [
                            chain.get("Name", "")
                            for chain in (item.get("BusinessChains") or [])
                            if chain.get("Name")
                        ]
                        category_label = ", ".join(
                            category.get("LocalizedName") or category.get("Name", "")
                            for category in categories
                            if category.get("LocalizedName") or category.get("Name")
                        )
                        store = {
                            "name": title,
                            "address": addr_label,
                            "distanceMeters": dist_m,
                            "lat": store_lat,
                            "lng": store_lng,
                            "category": category_label,
                            "businessChains": chain_names,
                        }
                        address_key = normalize(addr_label)
                        normalized_title = normalize(title)
                        known_brands = (
                            "coopmart", "coop food", "winmart", "bach hoa xanh",
                            "kingfoodmart", "tops market", "lotte mart", "mega market",
                            "emart", "satrafood",
                        )
                        brand_key = next((brand for brand in known_brands if brand in normalized_title), None)
                        place_id = item.get("PlaceId")
                        key = (brand_key, address_key) if brand_key else (
                            place_id or (normalized_title, address_key)
                        )
                        stores_by_key[key] = store
                except Exception:
                    logger.exception("SearchText fallback failed for query=%r", search_term)

        def budget_group(store):
            name = normalize(store.get("name", ""))

            if (
                "bach hoa xanh" in name
                or "winmart+" in name
                or "win mart+" in name
                or "coop food" in name
                or "co.op food" in name
                or "satrafood" in name
                or "cua hang thuc pham" in name
            ):
                return "mid", "Tầm trung (35.000đ – 70.000đ)"

            if (
                "coopmart" in name or "co.opmart" in name
                or ("winmart" in name and "+" not in name)
                or ("win mart" in name and "+" not in name)
                or "kingfood" in name or "tops market" in name
                or "lotte mart" in name or "mega market" in name
                or "emart" in name or "go market" in name
                or "sieu thi" in name
            ):
                return "high", "Siêu thị lớn (trên 70.000đ)"

            if "cho " in name or "tap hoa" in name or "bach hoa" in name:
                return "low", "Tiết kiệm (dưới 35.000đ)"

            return "mid", "Tầm trung (35.000đ – 70.000đ)"

        stores = []
        for store in stores_by_key.values():
            group_key, group_label = budget_group(store)
            store["budgetGroup"] = group_key
            store["budgetLabel"] = group_label
            stores.append(store)

        if tier not in ("low", "mid", "high"):
            tier = "mid"
        selected_stores = [
            store for store in stores
            if store["budgetGroup"] == tier
        ]
        selected_stores.sort(key=lambda item: item["distanceMeters"])

        logger.info(
            "SearchNearby pages=%s stores_found=%s stores_selected=%s tier=%s location=(%s,%s)",
            page_count, len(stores), len(selected_stores), tier, lat, lng
        )
        return response(200, {
            "source": "amazon-location-search-nearby",
            "advice": advice,
            "selectedTier": tier,
            "origin": {"lat": lat, "lng": lng},
            "searchRadiusMeters": 10000,
            "stores": selected_stores[:30],
        })
    except Exception:
        logger.exception("Amazon Location SearchNearby failed")
        return response(200, {
            "source": "fallback",
            "advice": advice,
            "origin": {"lat": lat, "lng": lng},
            "stores": [],
        })


# 2. CHATBOT ĐẦU BẾP AI TÂM LÝ THEO TÂM TRẠNG & THỜI TIẾT (AMAZON BEDROCK)
def find_recipes_by_mood_weather(msg, items):
    norm_msg = normalize(msg)
    target_moods = []
    target_cats = []

    if any(k in norm_msg for k in ["mua", "lanh", "se lanh", "buon", "stress", "ap luc", "am bung", "co don"]):
        target_moods.append("comforting")
        target_cats.extend(["soup", "main"])
    if any(k in norm_msg for k in ["nong", "oi buc", "nang", "ngan", "nhe bung", "giam can", "thanh dam", "mat"]):
        target_moods.extend(["fresh", "light"])
        target_cats.extend(["salad", "side", "soup"])
    if any(k in norm_msg for k in ["met", "duoi", "ue oai", "can nang luong", "doi", "tap gym", "vui"]):
        target_moods.append("energizing")
        target_cats.append("main")

    direct = [
        r for r in items
        if normalize(r.get("Name", "")) in norm_msg
        or any(normalize(ing) in norm_msg for ing in (r.get("Ingredients") or []))
    ]
    if direct:
        random.shuffle(direct)
        return direct

    scored = []
    for r in items:
        score = 0
        r_moods = r.get("MoodTags") or []
        if any(m in r_moods for m in target_moods):
            score += 2
        if r.get("Category") in target_cats:
            score += 1
        if ("chay" in norm_msg or "thanh dam" in norm_msg) and "vegetarian" in (r.get("DietTags") or []):
            score += 3
        if score > 0:
            scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return []

    top_score = scored[0][0]
    top_group = [r for score, r in scored if score >= max(1, top_score - 1)]
    random.shuffle(top_group)
    rest_group = [r for score, r in scored if score < max(1, top_score - 1)]
    return top_group + rest_group


def handle_chat(event):
    payload = parse_json_body(event)
    user_message = str(payload.get("message") or "").strip()
    if not user_message:
        raise ValueError("Vui lòng nhập tâm trạng, thời tiết hoặc câu hỏi của bạn.")
    if len(user_message) > 500:
        raise ValueError("Câu hỏi tối đa 500 ký tự.")

    items = get_all_recipes()
    shuffled_items = list(items)
    random.shuffle(shuffled_items)

    exclude_ids = set(payload.get("excludeIds") or [])

    matched_recipes = find_recipes_by_mood_weather(user_message, items)
    fresh_matches = [r for r in matched_recipes if r.get("RecipeID") not in exclude_ids]
    candidate_recipes = fresh_matches if fresh_matches else matched_recipes

    suggested_id = candidate_recipes[0].get("RecipeID") if candidate_recipes else None
    suggested_name = candidate_recipes[0].get("Name") if candidate_recipes else None

    menu_context = "\n".join([
        f"- [{r.get('RecipeID')}] {r.get('Name')}: loại {r.get('Category')}, "
        f"tâm trạng phù hợp: {', '.join(r.get('MoodTags') or [])}, "
        f"chi phí ~{r.get('EstimatedCostVND')}đ, nấu {r.get('TotalTimeMinutes') or r.get('CookingTimeMinutes', 20)} phút, "
        f"nguyên liệu: {', '.join(r.get('Ingredients') or [])}"
        for r in shuffled_items
    ])

    avoid_note = ""
    if exclude_ids:
        avoid_names = [r.get("Name") for r in items if r.get("RecipeID") in exclude_ids]
        if avoid_names:
            avoid_note = f"\n4. KHÔNG gợi ý lại các món người dùng vừa xem ở lượt trước: {', '.join(avoid_names)}. Hãy chọn món khác trong danh sách."

    primary_hint = (
        f"2. Gợi ý món **{suggested_name}** (hoặc 1–2 món phù hợp khác TỪ DANH SÁCH BÊN DƯỚI, ghi rõ tên món chính xác, chi phí và thời gian nấu), giải thích ngắn gọn vì sao món đó hợp với tâm trạng/thời tiết hiện tại.\n"
        if suggested_name
        else "2. Gợi ý 1–2 món phù hợp nhất TỪ DANH SÁCH BÊN DƯỚI (ghi rõ tên món chính xác, chi phí và thời gian nấu), giải thích ngắn gọn vì sao món đó hợp với tâm trạng/thời tiết hiện tại.\n"
    )

    system_prompt = (
        "Bạn là Đầu Bếp AI Tâm Lý trên website hướng dẫn dạy nấu ăn Việt Nam. "
        "Người dùng có thể hỏi gợi ý món ăn theo TÂM TRẠNG (vui, buồn, stress, mệt mỏi, cần năng lượng, ăn thanh đạm), "
        "THỜI TIẾT (trời mưa, trời lạnh, nắng nóng, oi bức), ngân sách hoặc nguyên liệu sẵn có.\n"
        "Quy tắc trả lời:\n"
        "1. Đồng cảm nhẹ nhàng với tâm trạng hoặc thời tiết của người dùng (1 câu ngắn).\n"
        f"{primary_hint}"
        "3. Khích lệ người dùng bấm nút xem hướng dẫn từng bước bên dưới."
        f"{avoid_note}\n"
        "Trả lời bằng tiếng Việt tự nhiên, ấm áp, đa dạng cách diễn đạt, ngắn gọn (tối đa 5 câu).\n\n"
        f"Danh sách công thức hiện có:\n{menu_context}"
    )

    try:
        if BEDROCK_API_KEY:
            encoded_model = quote(BEDROCK_MODEL_ID, safe="")
            endpoint = f"https://bedrock-runtime.{BEDROCK_REGION}.amazonaws.com/model/{encoded_model}/converse"
            req_body = json.dumps({
                "messages": [{"role": "user", "content": [{"text": user_message}]}],
                "system": [{"text": system_prompt}],
                "inferenceConfig": {"maxTokens": 450, "temperature": 0.9},
            }).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=req_body,
                headers={
                    "Authorization": f"Bearer {BEDROCK_API_KEY}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                res = json.loads(resp.read().decode("utf-8"))
        else:
            res = bedrock_client.converse(
                modelId=BEDROCK_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": user_message}]}],
                system=[{"text": system_prompt}],
                inferenceConfig={"maxTokens": 450, "temperature": 0.9},
            )

        reply_text = res["output"]["message"]["content"][0]["text"]
        return response(200, {
            "reply": reply_text,
            "suggestedRecipeId": suggested_id,
            "suggestedRecipeName": suggested_name,
            "source": "amazon-bedrock",
        })
    except urllib.error.HTTPError as http_exc:
        err_detail = http_exc.read().decode("utf-8", errors="ignore")
        logger.warning("Amazon Bedrock HTTP %s: %s", http_exc.code, err_detail)
    except Exception as exc:
        logger.warning("Amazon Bedrock fallback triggered: %s", exc)

    if candidate_recipes:
        top = candidate_recipes[0]
        second = candidate_recipes[1] if len(candidate_recipes) > 1 else None
        extra = f" hoặc <strong>{second.get('Name')}</strong> (~{second.get('EstimatedCostVND')}đ)" if second else ""
        reply = (
            f"Với thời tiết và tâm trạng hiện tại của bạn, một bữa ăn tự nấu sẽ giúp tinh thần dễ chịu hơn nhiều! "
            f"Mình gợi ý món <strong>{top.get('Name')}</strong> (~{top.get('EstimatedCostVND')}đ, "
            f"{top.get('TotalTimeMinutes') or top.get('CookingTimeMinutes', 20)} phút){extra}. "
            f"Bạn bấm vào nút bên dưới để xem hướng dẫn chi tiết từng bước nhé!"
        )
    else:
        reply = (
            "Dù tâm trạng hay thời tiết hôm nay thế nào, tự vào bếp nấu một món ngon cũng giúp thư giãn rất tốt! "
            "Bạn hãy cho mình biết bạn đang thích ăn món nóng hổi (canh/kho) hay món thanh mát (salad/xào) để mình chọn công thức chuẩn nhất nhé."
        )
    return response(200, {
        "reply": reply,
        "suggestedRecipeId": suggested_id,
        "suggestedRecipeName": suggested_name,
        "source": "fallback",
    })


# 3. TÌM KIẾM CÔNG THỨC NẤU ĂN TỪ DYNAMODB
def handle_recipe_search(params, context):
    filters = {
        "ingredients": csv_values(params.get("ingredients", ""), "ingredients"),
        "allergies": csv_values(params.get("allergies", ""), "allergies"),
        "avoid": csv_values(params.get("avoid", ""), "avoid"),
        "health": (params.get("health") or "").strip(),
        "need": (params.get("need") or "").strip(),
        "mood": (params.get("mood") or "").strip(),
        "maxTime": None,
        "maxBudget": None,
        "maxSodium": None,
        "maxCarbs": None,
    }
    allowed_tags = {
        "health": {"low_sodium", "controlled_carb", "low_saturated_fat", "light", "high_protein"},
        "need": {"high_protein", "vegetarian", "vegan", "low_calorie"},
        "mood": {"comforting", "fresh", "energizing", "light"},
    }
    for key, allowed in allowed_tags.items():
        if filters[key] and filters[key] not in allowed:
            return response(400, {"error": f"Giá trị bộ lọc {key} không hợp lệ."})

    numeric_limits = {
        "maxTime": (600, 0),
        "maxBudget": (10000000, 0),
        "maxSodium": (100000, 0),
        "maxCarbs": (1000, 0),
    }
    for key, (upper, lower) in numeric_limits.items():
        raw = params.get(key)
        if raw not in (None, ""):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return response(400, {"error": f"Tham số {key} phải là số."})
            if value < lower or value > upper:
                return response(400, {"error": f"Tham số {key} nằm ngoài giới hạn cho phép."})
            filters[key] = value

    logger.info(
        "RequestId=%s filters=%s random=%s",
        getattr(context, "aws_request_id", "local"),
        filters,
        params.get("random") == "true",
    )
    recipes = get_all_recipes()
    matched = [item for item in recipes if matches(item, filters)]
    matched.sort(key=lambda item: ingredient_score(item, filters["ingredients"]), reverse=True)
    if params.get("random", "false").lower() == "true":
        random.shuffle(matched)
        matched = matched[:6]

    return response(200, {
        "recipes": matched,
        "count": len(matched),
        "filters": filters,
        "source": "dynamodb",
    })


def lambda_handler(event, context):
    method = (
        event.get("httpMethod")
        or (event.get("requestContext", {}).get("http") or {}).get("method")
        or "GET"
    ).upper()
    raw_path = event.get("rawPath") or event.get("path") or ""
    route_key = event.get("routeKey") or ""

    if method == "OPTIONS":
        return response(200, {"message": "CORS OK"})

    try:
        if method == "POST" and ("chat" in route_key or raw_path.endswith("/chat")):
            return handle_chat(event)

        if method == "GET" and ("nearby-stores" in route_key or raw_path.endswith("/nearby-stores")):
            return handle_nearby_stores(query_params(event))

        if method == "GET":
            return handle_recipe_search(query_params(event), context)

        return response(405, {"error": "Phương thức HTTP không được hỗ trợ."})
    except ValueError as exc:
        return response(400, {"error": str(exc)})
    except ClientError:
        logger.exception("AWS service request failed")
        return response(500, {"error": f"Không truy cập được dịch vụ AWS. Kiểm tra tên bảng {TABLE_NAME} và IAM Role."})
    except Exception:
        logger.exception("Unhandled Lambda error")
        return response(500, {"error": "Lỗi nội bộ. Xem CloudWatch Logs để biết chi tiết."})
