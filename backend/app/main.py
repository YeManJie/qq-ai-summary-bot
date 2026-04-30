from fastapi import FastAPI, Query
from pydantic import BaseModel
from datetime import datetime
import re
import json
import html
import traceback

from app.database import SessionLocal, Message, init_db
from app.ai.client import LLMClient
from app.ai.prompts import (
    SLICE_SYSTEM_PROMPT,
    build_slice_user_prompt,
    DENOISE_SYSTEM_PROMPT,
    build_denoise_user_prompt,
    CLUSTER_SYSTEM_PROMPT,
    build_cluster_user_prompt,
    CLASSIFY_SYSTEM_PROMPT,
    build_classify_user_prompt,
)

app = FastAPI()

init_db()


class ProcessPipelineRequest(BaseModel):
    group_id: str
    start_time: str
    end_time: str


def normalize_url(url: str | None) -> str | None:
    """统一规范化 URL，把 HTML 转义字符还原。"""
    if not url:
        return None
    return html.unescape(str(url)).strip()


def clean_message(raw_message: str) -> str:
    """去掉 CQ 码，只保留可读文本。"""
    if not raw_message:
        return ""

    cleaned = re.sub(r"\[CQ:[^\]]+\]", "", raw_message)
    return cleaned.strip()


def extract_reply_target(raw_message: str) -> str | None:
    """提取回复目标消息 ID。"""
    if not raw_message:
        return None

    match = re.search(r"\[CQ:reply,id=(\d+)", raw_message)
    if match:
        return match.group(1)
    return None


def extract_mentioned_users(raw_message: str) -> list[str]:
    """提取被 @ 的用户列表。"""
    if not raw_message:
        return []

    return re.findall(r"\[CQ:at,qq=(\d+)\]", raw_message)


def detect_message_features(raw_message: str, resource_info: dict) -> dict:
    """
    根据 raw_message 和资源信息，生成消息描述字段。
    优先把 animation 判成 animation，不让它轻易落入 mixed。
    """
    is_reply = "[CQ:reply" in (raw_message or "")
    reply_target_id = extract_reply_target(raw_message)

    mentioned_users = extract_mentioned_users(raw_message)
    has_at = len(mentioned_users) > 0

    has_image = len(resource_info["image_urls"]) > 0 or len(resource_info["image_files"]) > 0
    has_file = len(resource_info["file_urls"]) > 0 or len(resource_info["file_names"]) > 0
    has_face = len(resource_info["face_ids"]) > 0
    has_animation = len(resource_info["animation_urls"]) > 0 or len(resource_info["animation_files"]) > 0

    cleaned = clean_message(raw_message)

    if has_animation and not cleaned and not has_file and not has_face:
        message_type = "animation"
    elif has_image and not cleaned and not has_file and not has_face and not has_animation:
        message_type = "image"
    elif has_file and not cleaned and not has_image and not has_face and not has_animation:
        message_type = "file"
    elif has_face and not cleaned and not has_image and not has_file and not has_animation:
        message_type = "face"
    elif cleaned and not has_image and not has_file and not has_face and not has_animation:
        message_type = "text"
    else:
        message_type = "mixed"

    sub_type = "normal"
    if is_reply:
        sub_type = "reply"
    elif has_animation:
        sub_type = "animation_face"

    return {
        "message_type": message_type,
        "sub_type": sub_type,
        "is_reply": is_reply,
        "reply_target_id": reply_target_id,
        "has_at": has_at,
        "mentioned_users": mentioned_users,
        "has_image": has_image,
        "has_file": has_file,
        "has_face": has_face,
        "has_animation": has_animation,
    }


def extract_resources_from_segments(message_segments: list) -> dict:
    """
    优先从 NapCat 的结构化 message 段里提取资源。
    """
    image_urls: list[str] = []
    image_files: list[str] = []
    image_sizes: list[str] = []

    file_urls: list[str] = []
    file_names: list[str] = []
    file_sizes: list[str] = []

    face_ids: list[str] = []

    animation_files: list[str] = []
    animation_urls: list[str] = []

    resources: list[dict] = []

    for seg in message_segments or []:
        seg_type = seg.get("type")
        seg_data = seg.get("data", {})

        if seg_type == "image":
            url = normalize_url(seg_data.get("url"))
            file_name = seg_data.get("file")
            file_size = seg_data.get("file_size")
            sub_type = str(seg_data.get("sub_type")) if seg_data.get("sub_type") is not None else None

            is_animation = sub_type == "1"

            if is_animation:
                if file_name:
                    animation_files.append(str(file_name))
                if url:
                    animation_urls.append(url)

                resources.append({
                    "type": "animation",
                    "url": url,
                    "file": str(file_name) if file_name is not None else None,
                    "file_size": str(file_size) if file_size is not None else None,
                    "sub_type": sub_type,
                    "raw_data": seg_data
                })
            else:
                if url:
                    image_urls.append(url)
                if file_name:
                    image_files.append(str(file_name))
                if file_size is not None:
                    image_sizes.append(str(file_size))

                resources.append({
                    "type": "image",
                    "url": url,
                    "file": str(file_name) if file_name is not None else None,
                    "file_size": str(file_size) if file_size is not None else None,
                    "sub_type": sub_type,
                    "raw_data": seg_data
                })

        elif seg_type == "file":
            url = normalize_url(seg_data.get("url"))
            file_name = seg_data.get("name") or seg_data.get("file")
            file_size = seg_data.get("file_size")

            if url:
                file_urls.append(url)
            if file_name:
                file_names.append(str(file_name))
            if file_size is not None:
                file_sizes.append(str(file_size))

            resources.append({
                "type": "file",
                "url": url,
                "name": str(file_name) if file_name is not None else None,
                "file_size": str(file_size) if file_size is not None else None,
                "raw_data": seg_data
            })

        elif seg_type == "face":
            face_id = seg_data.get("id")
            if face_id is not None:
                face_ids.append(str(face_id))

            resources.append({
                "type": "face",
                "id": str(face_id) if face_id is not None else None,
                "raw_data": seg_data
            })

    return {
        "image_urls": image_urls,
        "image_files": image_files,
        "image_sizes": image_sizes,
        "file_urls": file_urls,
        "file_names": file_names,
        "file_sizes": file_sizes,
        "face_ids": face_ids,
        "animation_files": animation_files,
        "animation_urls": animation_urls,
        "resources": resources,
    }


def extract_resources_from_raw_message(raw_message: str) -> dict:
    """
    从 raw_message 里兜底提取资源。
    """
    image_urls: list[str] = []
    image_files: list[str] = []
    image_sizes: list[str] = []

    file_urls: list[str] = []
    file_names: list[str] = []
    file_sizes: list[str] = []

    face_ids: list[str] = []

    animation_files: list[str] = []
    animation_urls: list[str] = []

    resources: list[dict] = []

    if not raw_message:
        return {
            "image_urls": image_urls,
            "image_files": image_files,
            "image_sizes": image_sizes,
            "file_urls": file_urls,
            "file_names": file_names,
            "file_sizes": file_sizes,
            "face_ids": face_ids,
            "animation_files": animation_files,
            "animation_urls": animation_urls,
            "resources": resources,
        }

    for match in re.finditer(r"\[CQ:face,([^\]]+)\]", raw_message):
        seg_text = html.unescape(match.group(1))
        face_id_match = re.search(r"id=(\d+)", seg_text)
        if face_id_match:
            face_id = face_id_match.group(1)
            face_ids.append(face_id)
            resources.append({
                "type": "face",
                "id": face_id,
                "raw_segment": html.unescape(match.group(0))
            })

    for match in re.finditer(r"\[CQ:image,([^\]]+)\]", raw_message):
        seg_text = html.unescape(match.group(1))

        file_match = re.search(r"file=([^,]+)", seg_text)
        url_match = re.search(r"url=([^,\]]+)", seg_text)
        sub_type_match = re.search(r"sub_type=([^,]+)", seg_text)
        size_match = re.search(r"file_size=(\d+)", seg_text)

        file_name = file_match.group(1) if file_match else None
        url = normalize_url(url_match.group(1)) if url_match else None
        sub_type = sub_type_match.group(1) if sub_type_match else None
        file_size = size_match.group(1) if size_match else None

        is_animation = sub_type == "1"

        if is_animation:
            if file_name:
                animation_files.append(file_name)
            if url:
                animation_urls.append(url)

            resources.append({
                "type": "animation",
                "file": file_name,
                "url": url,
                "sub_type": sub_type,
                "file_size": file_size,
                "raw_segment": html.unescape(match.group(0))
            })
        else:
            if file_name:
                image_files.append(file_name)
            if url:
                image_urls.append(url)
            if file_size:
                image_sizes.append(file_size)

            resources.append({
                "type": "image",
                "file": file_name,
                "url": url,
                "sub_type": sub_type,
                "file_size": file_size,
                "raw_segment": html.unescape(match.group(0))
            })

    for match in re.finditer(r"\[CQ:file,([^\]]+)\]", raw_message):
        seg_text = html.unescape(match.group(1))

        file_match = re.search(r"file=([^,]+)", seg_text)
        url_match = re.search(r"url=([^,\]]+)", seg_text)
        name_match = re.search(r"name=([^,]+)", seg_text)
        size_match = re.search(r"file_size=(\d+)", seg_text)

        file_name = name_match.group(1) if name_match else None
        if not file_name and file_match:
            file_name = file_match.group(1)

        url = normalize_url(url_match.group(1)) if url_match else None
        file_size = size_match.group(1) if size_match else None

        if url:
            file_urls.append(url)
        if file_name:
            file_names.append(file_name)
        if file_size:
            file_sizes.append(file_size)

        resources.append({
            "type": "file",
            "name": file_name,
            "url": url,
            "file_size": file_size,
            "raw_segment": html.unescape(match.group(0))
        })

    return {
        "image_urls": image_urls,
        "image_files": image_files,
        "image_sizes": image_sizes,
        "file_urls": file_urls,
        "file_names": file_names,
        "file_sizes": file_sizes,
        "face_ids": face_ids,
        "animation_files": animation_files,
        "animation_urls": animation_urls,
        "resources": resources,
    }


def dedupe_resource_objects(resources: list[dict]) -> list[dict]:
    """
    对 resource_json 做对象级去重。
    """
    deduped = []
    seen = set()

    for item in resources:
        item_type = item.get("type")
        url = normalize_url(item.get("url"))
        file_value = item.get("file")
        name_value = item.get("name")
        id_value = item.get("id")

        if item_type == "face":
            key = ("face", id_value)
        elif item_type in ("image", "animation"):
            key = (item_type, url, file_value)
        elif item_type == "file":
            key = ("file", url, name_value)
        else:
            key = (item_type, url, file_value, name_value, id_value)

        if key in seen:
            continue
        seen.add(key)

        new_item = dict(item)
        if "url" in new_item:
            new_item["url"] = url

        deduped.append(new_item)

    return deduped


def merge_resource_info(segment_info: dict, raw_info: dict) -> dict:
    """
    合并结构化 message 段提取结果和 raw_message 兜底提取结果。
    """
    def unique_list(values):
        result = []
        for v in values:
            if v is None:
                continue
            v = str(v)

            if v.startswith("http://") or v.startswith("https://"):
                v = normalize_url(v)

            if v not in result:
                result.append(v)
        return result

    merged_resources = dedupe_resource_objects(
        segment_info["resources"] + raw_info["resources"]
    )

    return {
        "image_urls": unique_list(segment_info["image_urls"] + raw_info["image_urls"]),
        "image_files": unique_list(segment_info["image_files"] + raw_info["image_files"]),
        "image_sizes": unique_list(segment_info["image_sizes"] + raw_info["image_sizes"]),
        "file_urls": unique_list(segment_info["file_urls"] + raw_info["file_urls"]),
        "file_names": unique_list(segment_info["file_names"] + raw_info["file_names"]),
        "file_sizes": unique_list(segment_info["file_sizes"] + raw_info["file_sizes"]),
        "face_ids": unique_list(segment_info["face_ids"] + raw_info["face_ids"]),
        "animation_files": unique_list(segment_info["animation_files"] + raw_info["animation_files"]),
        "animation_urls": unique_list(segment_info["animation_urls"] + raw_info["animation_urls"]),
        "resources": merged_resources,
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/napcat/callback")
async def napcat_callback(data: dict):
    if data.get("post_type") != "message":
        return {"ok": True, "message": "不是消息事件，已忽略"}

    if data.get("message_type") != "group":
        return {"ok": True, "message": "不是群消息，已忽略"}

    group_id = str(data.get("group_id"))
    user_id = str(data.get("user_id"))
    message_id = str(data.get("message_id"))
    raw_message = data.get("raw_message") or ""
    message_segments = data.get("message", [])

    cleaned_message = clean_message(raw_message)
    received_at = datetime.now().isoformat()

    segment_resource_info = extract_resources_from_segments(message_segments)
    raw_resource_info = extract_resources_from_raw_message(raw_message)
    resource_info = merge_resource_info(segment_resource_info, raw_resource_info)

    features = detect_message_features(raw_message, resource_info)

    db = SessionLocal()

    try:
        message = Message(
            group_id=group_id,
            user_id=user_id,
            message_id=message_id,
            raw_message=raw_message,
            cleaned_message=cleaned_message,
            message_type=features["message_type"],
            sub_type=features["sub_type"],
            is_reply=features["is_reply"],
            reply_target_id=features["reply_target_id"],
            has_at=features["has_at"],
            mentioned_users=json.dumps(features["mentioned_users"], ensure_ascii=False),
            has_image=features["has_image"],
            has_file=features["has_file"],
            has_face=features["has_face"],
            has_animation=features["has_animation"],
            image_urls=json.dumps(resource_info["image_urls"], ensure_ascii=False),
            image_files=json.dumps(resource_info["image_files"], ensure_ascii=False),
            image_sizes=json.dumps(resource_info["image_sizes"], ensure_ascii=False),
            file_urls=json.dumps(resource_info["file_urls"], ensure_ascii=False),
            file_names=json.dumps(resource_info["file_names"], ensure_ascii=False),
            file_sizes=json.dumps(resource_info["file_sizes"], ensure_ascii=False),
            face_ids=json.dumps(resource_info["face_ids"], ensure_ascii=False),
            animation_files=json.dumps(resource_info["animation_files"], ensure_ascii=False),
            animation_urls=json.dumps(resource_info["animation_urls"], ensure_ascii=False),
            resource_json=json.dumps(resource_info["resources"], ensure_ascii=False),
            message_segments_json=json.dumps(message_segments, ensure_ascii=False),
            received_at=received_at
        )

        db.add(message)
        db.commit()
        db.refresh(message)

        return {
            "ok": True,
            "saved": True,
            "id": message.id,
            "group_id": group_id,
            "user_id": user_id,
            "message_id": message_id,
            "raw_message": raw_message,
            "cleaned_message": cleaned_message,
            "message_type": message.message_type,
            "sub_type": message.sub_type,
            "is_reply": message.is_reply,
            "reply_target_id": message.reply_target_id,
            "has_at": message.has_at,
            "mentioned_users": features["mentioned_users"],
            "has_image": message.has_image,
            "has_file": message.has_file,
            "has_face": message.has_face,
            "has_animation": message.has_animation,
            "image_urls": resource_info["image_urls"],
            "file_urls": resource_info["file_urls"],
            "face_ids": resource_info["face_ids"],
            "animation_urls": resource_info["animation_urls"],
            "received_at": received_at
        }
    finally:
        db.close()


@app.get("/messages")
def get_messages(
    group_id: str | None = Query(default=None),
    keyword: str | None = Query(default=None),
    start_time: str | None = Query(default=None),
    end_time: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100)
):
    db = SessionLocal()

    try:
        query = db.query(Message)

        if group_id:
            query = query.filter(Message.group_id == group_id)

        if keyword:
            query = query.filter(Message.cleaned_message.contains(keyword))

        if start_time:
            query = query.filter(Message.received_at >= start_time)

        if end_time:
            query = query.filter(Message.received_at <= end_time)

        total = query.count()
        offset = (page - 1) * page_size

        messages = (
            query.order_by(Message.id.desc())
            .offset(offset)
            .limit(page_size)
            .all()
        )

        items = []
        for message in messages:
            items.append({
                "id": message.id,
                "group_id": message.group_id,
                "user_id": message.user_id,
                "message_id": message.message_id,
                "raw_message": message.raw_message,
                "cleaned_message": message.cleaned_message,
                "message_type": message.message_type,
                "sub_type": message.sub_type,
                "is_reply": message.is_reply,
                "reply_target_id": message.reply_target_id,
                "has_at": message.has_at,
                "mentioned_users": message.mentioned_users,
                "has_image": message.has_image,
                "has_file": message.has_file,
                "has_face": message.has_face,
                "has_animation": message.has_animation,
                "image_urls": message.image_urls,
                "image_files": message.image_files,
                "image_sizes": message.image_sizes,
                "file_urls": message.file_urls,
                "file_names": message.file_names,
                "file_sizes": message.file_sizes,
                "face_ids": message.face_ids,
                "animation_files": message.animation_files,
                "animation_urls": message.animation_urls,
                "resource_json": message.resource_json,
                "message_segments_json": message.message_segments_json,
                "received_at": message.received_at
            })

        return {
            "page": page,
            "page_size": page_size,
            "total": total,
            "items": items
        }
    finally:
        db.close()


@app.post("/process/pipeline")
def process_pipeline(request: ProcessPipelineRequest):
    """
    第二阶段最小原型：
    1. 按 group_id + 时间窗口取消息
    2. 调用 LLM 做 AI 切片
    3. 调用 LLM 做切片级降噪
    4. 调用 LLM 做聚合，得到对话簇
    5. 调用 LLM 做对话簇分类
    """
    db = SessionLocal()

    try:
        messages = (
            db.query(Message)
            .filter(Message.group_id == request.group_id)
            .filter(Message.received_at >= request.start_time)
            .filter(Message.received_at <= request.end_time)
            .order_by(Message.received_at.asc())
            .all()
        )

        print("step 1: query messages ok, count =", len(messages))

        messages_for_ai = []
        for message in messages:
            messages_for_ai.append({
                "message_id": message.message_id,
                "time": message.received_at,
                "user_id": message.user_id,
                "cleaned_message": message.cleaned_message or "",
                "message_type": message.message_type,
                "sub_type": message.sub_type,
                "is_reply": message.is_reply,
                "reply_target_id": message.reply_target_id,
                "has_at": message.has_at,
                "mentioned_users": json.loads(message.mentioned_users) if message.mentioned_users else [],
                "has_file": message.has_file,
                "has_image": message.has_image,
                "has_face": message.has_face,
                "has_animation": message.has_animation,
                "image_urls": json.loads(message.image_urls) if message.image_urls else [],
                "image_files": json.loads(message.image_files) if message.image_files else [],
                "image_sizes": json.loads(message.image_sizes) if message.image_sizes else [],
                "file_urls": json.loads(message.file_urls) if message.file_urls else [],
                "file_names": json.loads(message.file_names) if message.file_names else [],
                "file_sizes": json.loads(message.file_sizes) if message.file_sizes else [],
                "face_ids": json.loads(message.face_ids) if message.face_ids else [],
                "animation_files": json.loads(message.animation_files) if message.animation_files else [],
                "animation_urls": json.loads(message.animation_urls) if message.animation_urls else [],
                "resource_json": json.loads(message.resource_json) if message.resource_json else [],
            })

        print("step 2: messages_for_ai built")

        if not messages_for_ai:
            return {
                "ok": True,
                "group_id": request.group_id,
                "start_time": request.start_time,
                "end_time": request.end_time,
                "count": 0,
                "messages_for_ai": [],
                "slice_result": None,
                "denoise_result": None,
                "cluster_result": None,
                "classification_result": None,
            }

        client = LLMClient()
        print("step 3: client created")

        # 第一步：AI 切片
        slice_user_prompt = build_slice_user_prompt(messages_for_ai)
        print("step 4: slice prompt built, length =", len(slice_user_prompt))

        slice_result = client.chat(
            system_prompt=SLICE_SYSTEM_PROMPT,
            user_prompt=slice_user_prompt,
            temperature=0.2
        )
        print("step 5: slice done")

        parsed_slice_json = slice_result.get("parsed_json")
        slices = []

        if isinstance(parsed_slice_json, dict):
            slices = parsed_slice_json.get("slices", [])

        print("step 6: slices parsed, count =", len(slices))

        # 第二步：切片级 AI 降噪
        denoise_result = None
        slice_noise_results = []

        if slices:
            denoise_user_prompt = build_denoise_user_prompt(messages_for_ai, slices)
            print("step 7: denoise prompt built, length =", len(denoise_user_prompt))

            denoise_result = client.chat(
                system_prompt=DENOISE_SYSTEM_PROMPT,
                user_prompt=denoise_user_prompt,
                temperature=0.2
            )
            print("step 8: denoise done")

            parsed_denoise_json = denoise_result.get("parsed_json")
            if isinstance(parsed_denoise_json, dict):
                slice_noise_results = parsed_denoise_json.get("slice_noise_results", [])

        print("step 9: slice_noise_results count =", len(slice_noise_results))

        # 第三步：AI 聚合
        cluster_result = None
        clusters = []

        if slices and slice_noise_results:
            cluster_user_prompt = build_cluster_user_prompt(
                messages_for_ai=messages_for_ai,
                slices=slices,
                slice_noise_results=slice_noise_results
            )
            print("step 10: cluster prompt built, length =", len(cluster_user_prompt))

            cluster_result = client.chat(
                system_prompt=CLUSTER_SYSTEM_PROMPT,
                user_prompt=cluster_user_prompt,
                temperature=0.2
            )
            print("step 11: cluster done")

            parsed_cluster_json = cluster_result.get("parsed_json")
            if isinstance(parsed_cluster_json, dict):
                clusters = parsed_cluster_json.get("clusters", [])

        print("step 12: clusters count =", len(clusters))

        # 第四步：AI 分类
        classification_result = None
        if clusters:
            classify_user_prompt = build_classify_user_prompt(
                messages_for_ai=messages_for_ai,
                clusters=clusters
            )
            print("step 13: classify prompt built, length =", len(classify_user_prompt))

            classification_result = client.chat(
                system_prompt=CLASSIFY_SYSTEM_PROMPT,
                user_prompt=classify_user_prompt,
                temperature=0.2
            )
            print("step 14: classify done")

        return {
            "ok": True,
            "group_id": request.group_id,
            "start_time": request.start_time,
            "end_time": request.end_time,
            "count": len(messages_for_ai),
            "messages_for_ai": messages_for_ai,
            "slice_result": slice_result,
            "denoise_result": denoise_result,
            "cluster_result": cluster_result,
            "classification_result": classification_result,
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "ok": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

    finally:
        db.close()