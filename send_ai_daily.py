#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 日报自动化系统
- 从 RSS 源抓取最近 48 小时内容
- 使用大模型 API 评分并生成日报（支持 OpenAI / 通义千问）
- 发送到飞书群（自定义机器人 + 签名校验）
- 基于 sha256(link) 去重
"""

import os
import sys
import json
import hashlib
import hmac
import base64
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict

import requests
import feedparser
from dateutil import parser as date_parser


# ==================== 配置 ====================
# LLM 配置（支持 OpenAI 或通义千问）
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # openai 或 qwen
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")

# 飞书配置
FEISHU_WEBHOOK_URL = os.getenv("FEISHU_WEBHOOK_URL", "")
FEISHU_SECRET = os.getenv("FEISHU_SECRET", "")

# RSS 配置
RSS_URLS_RAW = os.getenv("RSS_URLS", "")

RSS_URLS = [line.strip() for line in RSS_URLS_RAW.strip().split("\n") if line.strip()]
SENT_HASHES_FILE = Path("data/sent_hashes.txt")
MAX_CANDIDATES = 30
TOP_N = 3
HOURS_WINDOW = 48


# ==================== 工具函数 ====================
def load_sent_hashes() -> set:
    """加载已发送的 hash 集合"""
    if not SENT_HASHES_FILE.exists():
        SENT_HASHES_FILE.parent.mkdir(parents=True, exist_ok=True)
        SENT_HASHES_FILE.touch()
        return set()
    with open(SENT_HASHES_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def save_sent_hashes(hashes: set):
    """保存已发送的 hash 集合"""
    SENT_HASHES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SENT_HASHES_FILE, "w", encoding="utf-8") as f:
        for h in sorted(hashes):
            f.write(h + "\n")


def hash_link(link: str) -> str:
    """计算链接的 sha256 hash"""
    return hashlib.sha256(link.encode("utf-8")).hexdigest()


def is_recent(published_str: str, hours: int = HOURS_WINDOW) -> bool:
    """判断条目是否在最近 N 小时内"""
    try:
        pub_time = date_parser.parse(published_str)
        if pub_time.tzinfo is None:
            pub_time = pub_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - pub_time) <= timedelta(hours=hours)
    except Exception:
        return False


# ==================== RSS 抓取 ====================
def fetch_rss_entries() -> List[Dict]:
    """从所有 RSS 源抓取最近 48 小时内的条目"""
    candidates = []
    sent_hashes = load_sent_hashes()

    for url in RSS_URLS:
        try:
            print(f"[INFO] 抓取 RSS: {url}")
            feed = feedparser.parse(url)
            for entry in feed.entries:
                link = entry.get("link", "")
                if not link:
                    continue
                link_hash = hash_link(link)
                if link_hash in sent_hashes:
                    continue
                published = entry.get("published", entry.get("updated", ""))
                if not is_recent(published, HOURS_WINDOW):
                    continue
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                if len(summary) > 500:
                    summary = summary[:500] + "..."
                candidates.append({
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "published": published,
                    "hash": link_hash
                })
                if len(candidates) >= MAX_CANDIDATES:
                    break
        except Exception as e:
            print(f"[WARN] 抓取 {url} 失败: {e}")
            continue
        if len(candidates) >= MAX_CANDIDATES:
            break

    print(f"[INFO] 共收集 {len(candidates)} 条候选")
    return candidates


# ==================== LLM API 调用 ====================
def call_llm_json(system_prompt: str, user_prompt: str) -> Dict:
    """调用大模型 API，要求返回 JSON（支持 OpenAI / 通义千问）"""
    if LLM_PROVIDER == "qwen":
        return call_qwen_json(system_prompt, user_prompt)
    else:
        return call_openai_json(system_prompt, user_prompt)


def call_openai_json(system_prompt: str, user_prompt: str, model: str = "gpt-4o-2024-08-06") -> Dict:
    """调用 OpenAI API，要求返回 JSON"""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        print(f"[ERROR] OpenAI API 调用失败: {e}")
        sys.exit(1)


def call_qwen_json(system_prompt: str, user_prompt: str) -> Dict:
    """调用通义千问 API，要求返回 JSON"""
    # 检查必要的配置
    if not DASHSCOPE_API_KEY:
        print("[ERROR] 未配置 DASHSCOPE_API_KEY")
        sys.exit(1)

    model = QWEN_MODEL if QWEN_MODEL else "qwen-plus"
    print(f"[INFO] 使用通义千问模型: {model}")

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }

    # 合并 system 和 user prompt
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": combined_prompt}
        ],
        "response_format": {"type": "json_object"}
    }

    try:
        print(f"[DEBUG] 请求 URL: {url}")
        print(f"[DEBUG] 请求 model: {payload['model']}")
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        print(f"[DEBUG] 响应状态码: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] 通义千问 API HTTP 错误: {e}")
        print(f"[DEBUG] 响应内容: {resp.text}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 通义千问 API 调用失败: {e}")
        print(f"[DEBUG] 响应内容: {resp.text if 'resp' in locals() else 'No response'}")
        sys.exit(1)


# ==================== 评分阶段 ====================
def score_entries(entries: List[Dict]) -> List[Dict]:
    """使用 LLM 对条目打分，返回 Top N"""
    if not entries:
        return []

    system_prompt = """你是一名资深 AI 工程师和技术编辑。
你的任务是从一批 RSS 条目中，筛选出最值得企业内部 AI 团队关注的内容。
评分标准（0~10）：
- 大模型 / AI 平台能力更新：9~10
- Agent / Tool / RAG / 系统设计实践：7~9
- 产品应用案例、评测：5~7
- 泛泛而谈、营销软文：0~3

请返回 JSON 数组，每个元素包含：link, score, reason。
只返回 JSON，不要其他文字。"""

    user_prompt = f"""请对以下 {len(entries)} 条 RSS 条目打分：

{json.dumps(entries, ensure_ascii=False, indent=2)}

返回格式：
{{
  "scores": [
    {{"link": "...", "score": 8.5, "reason": "..."}},
    ...
  ]
}}"""

    result = call_llm_json(system_prompt, user_prompt)
    scores = result.get("scores", [])

    # 按 score 排序，取 Top N
    scores.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_scores = scores[:TOP_N]

    # 补充完整条目信息
    link_map = {e["link"]: e for e in entries}
    top_entries = []
    for s in top_scores:
        link = s["link"]
        if link in link_map:
            entry = link_map[link].copy()
            entry["score"] = s["score"]
            entry["score_reason"] = s["reason"]
            top_entries.append(entry)

    print(f"[INFO] 评分完成，Top {TOP_N}: {len(top_entries)} 条")
    return top_entries


# ==================== 日报生成阶段 ====================
def generate_daily_report(top_entries: List[Dict]) -> Dict:
    """使用 LLM 生成日报内容"""
    system_prompt = """你是企业内部"AI 日报"总编辑。读者是混合团队：老板、市场总监、项目经理、售前、算法、前端、后端、UI、测试、测绘。
你必须输出严格 JSON（不要 markdown、不要解释、不要多余文本）。
写作风格：少废话、强结论、可行动；禁止营销语、禁止感叹号、禁止"可关注/有一定帮助"等空话。
长度目标：整体内容约 200~280 个中文字符（不含 URL）。"""

    today = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")

    user_prompt = f"""基于以下 Top 3 RSS 条目，生成一张"终版 AI 日报卡片"的 JSON（中文），结构必须完全符合下面的 JSON 契约。

【Top 3 条目】
{json.dumps(top_entries, ensure_ascii=False, indent=2)}

【JSON 契约】
{{
  "date": "{today}",
  "theme": "今日主题一句话（发生了什么 + 为什么重要）",
  "decision": {{
    "level": "值得关注|需要试点|暂不行动",
    "reason": "≤15字原因"
  }},
  "core_changes": ["1条为佳，最多2条（主角变化，非背景）"],
  "related": ["0-2条可选（补充信息）"],
  "impacts": {{
    "business": {{
      "boss": "一句话判断价值/风险",
      "market": "一句话影响对外叙事/方案",
      "pm": "一句话影响交付/评估"
    }},
    "tech": {{
      "algo": "一句话影响模型/Agent设计",
      "frontend": "一句话影响交互/展示",
      "backend": "一句话影响架构/日志/成本",
      "qa": "一句话影响测试/定位"
    }},
    "delivery": {{
      "ui": "一句话影响设计依据/反馈",
      "presales": "一句话影响方案可信度/说服力",
      "surveying": "一句话影响标注/质检/交付透明度"
    }}
  }},
  "action": {{
    "label": "🧪建议试点|👀持续观察|❌可忽略",
    "detail": "建议：由谁在什么场景验证什么"
  }},
  "sources": [{{"title": "...", "link": "..."}}]
}}

【硬约束】
- impacts 每个字段都必须有内容；如果暂无明显影响，写"暂无明显影响"
- core_changes 不要超过2条；related 允许为空数组
- sources 必须来自 Top 3 条目的 title/link，最多3条
- 不允许出现"可关注/有一定帮助/值得一提"等空句
- theme 要体现"变化本身 + 业务价值"，不要泛泛而谈"""

    report = call_llm_json(system_prompt, user_prompt)
    return validate_and_fix_report(report)


def validate_and_fix_report(report: Dict) -> Dict:
    """校验并修复日报 JSON 结构，确保字段完整"""
    # 默认值
    default_report = {
        "date": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d"),
        "theme": "AI 技术动态",
        "decision": {"level": "持续观察", "reason": "待进一步评估"},
        "core_changes": [],
        "related": [],
        "impacts": {
            "business": {
                "boss": "暂无明显影响",
                "market": "暂无明显影响",
                "pm": "暂无明显影响"
            },
            "tech": {
                "algo": "暂无明显影响",
                "frontend": "暂无明显影响",
                "backend": "暂无明显影响",
                "qa": "暂无明显影响"
            },
            "delivery": {
                "ui": "暂无明显影响",
                "presales": "暂无明显影响",
                "surveying": "暂无明显影响"
            }
        },
        "action": {"label": "👀持续观察", "detail": "建议：持续关注相关动态"},
        "sources": []
    }

    # 合并默认值
    for key, default_value in default_report.items():
        if key not in report:
            report[key] = default_value
            print(f"[WARN] 缺失字段 {key}，使用默认值")

    # 校验并修复 impacts
    if "impacts" in report:
        for group in ["business", "tech", "delivery"]:
            if group not in report["impacts"]:
                report["impacts"][group] = default_report["impacts"][group]
            else:
                for role, default_text in default_report["impacts"][group].items():
                    if role not in report["impacts"][group] or not report["impacts"][group][role]:
                        report["impacts"][group][role] = default_text

    # 校验并修复 decision
    if "decision" not in report or not isinstance(report["decision"], dict):
        report["decision"] = default_report["decision"]
    else:
        if "level" not in report["decision"]:
            report["decision"]["level"] = "持续观察"
        if "reason" not in report["decision"]:
            report["decision"]["reason"] = "待进一步评估"

    # 截断 core_changes（最多2条）
    if "core_changes" in report and isinstance(report["core_changes"], list):
        report["core_changes"] = report["core_changes"][:2]

    # 截断 related（最多2条）
    if "related" in report and isinstance(report["related"], list):
        report["related"] = report["related"][:2]
    else:
        report["related"] = []

    # 截断 sources（最多3条）
    if "sources" in report and isinstance(report["sources"], list):
        report["sources"] = report["sources"][:3]

    # 校验 action
    if "action" not in report or not isinstance(report["action"], dict):
        report["action"] = default_report["action"]
    else:
        if "label" not in report["action"]:
            report["action"]["label"] = "👀持续观察"
        if "detail" not in report["action"]:
            report["action"]["detail"] = "建议：持续关注相关动态"

    print(f"[INFO] 日报结构校验完成")
    return report


# ==================== 飞书推送 ====================
def send_to_feishu(report: Dict):
    """发送日报到飞书群（带签名校验）"""
    if not FEISHU_WEBHOOK_URL:
        print("[WARN] 未配置 FEISHU_WEBHOOK_URL，跳过发送")
        return

    # 签名
    timestamp = str(int(time.time()))
    sign = ""
    if FEISHU_SECRET:
        string_to_sign = f"{timestamp}\n{FEISHU_SECRET}"
        hmac_code = hmac.new(
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256
        ).digest()
        sign = base64.b64encode(hmac_code).decode("utf-8")

    # 构建 impacts 分组列表（按终版模板：业务/技术/交付）
    impacts = report.get("impacts", {})

    # 🎯 业务 / 决策层
    business_impacts = impacts.get("business", {})
    business_text = f"**老板**: {business_impacts.get('boss', '暂无明显影响')}\n"
    business_text += f"**市场**: {business_impacts.get('market', '暂无明显影响')}\n"
    business_text += f"**产品经理**: {business_impacts.get('pm', '暂无明显影响')}"

    # 🧠 技术实现层
    tech_impacts = impacts.get("tech", {})
    tech_text = f"**算法工程师**: {tech_impacts.get('algo', '暂无明显影响')}\n"
    tech_text += f"**前端工程师**: {tech_impacts.get('frontend', '暂无明显影响')}\n"
    tech_text += f"**后端工程师**: {tech_impacts.get('backend', '暂无明显影响')}\n"
    tech_text += f"**测试工程师**: {tech_impacts.get('qa', '暂无明显影响')}"

    # 🎨 体验与交付层
    delivery_impacts = impacts.get("delivery", {})
    delivery_text = f"**UI设计师**: {delivery_impacts.get('ui', '暂无明显影响')}\n"
    delivery_text += f"**售前**: {delivery_impacts.get('presales', '暂无明显影响')}\n"
    delivery_text += f"**项目经理**: {delivery_impacts.get('surveying', '暂无明显影响')}"

    # 构建 core_changes 列表
    core_changes = report.get("core_changes", [])
    changes_text = "\n".join([f"• {c}" for c in core_changes]) if core_changes else "暂无"

    # 构建 related 列表
    related = report.get("related", [])
    related_text = "\n".join([f"• {r}" for r in related]) if related else ""

    # 构建 sources 列表
    sources = report.get("sources", [])
    sources_text = ""
    for idx, src in enumerate(sources, 1):
        sources_text += f"{idx}. [{src.get('title', '未知来源')}]({src.get('link', '#')})\n"
    if not sources_text:
        sources_text = "暂无来源"

    # 获取决策提示
    decision = report.get("decision", {})
    decision_level = decision.get("level", "持续观察")
    decision_reason = decision.get("reason", "待进一步评估")

    # 获取行动建议
    action = report.get("action", {})
    action_label = action.get("label", "👀持续观察")
    action_detail = action.get("detail", "建议：持续关注相关动态")

    # 构建卡片元素列表
    elements = [
        # 今日主题
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"**📌 今日主题**\n{report.get('theme', 'AI 技术动态')}"}
        },
        {"tag": "hr"},
        # 决策提示
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"**⚡ 决策提示**: {decision_level}\n💡 {decision_reason}"}
        },
        {"tag": "hr"},
        # 核心技术变化
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"**🔧 核心技术变化**\n{changes_text}"}
        }
    ]

    # 相关补充（可选）
    if related_text:
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"**📎 相关补充**\n{related_text}"}
        })

    # 角色影响速览（分组）
    elements.extend([
        {"tag": "hr"},
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"**👥 角色影响速览**\n\n🎯 **业务 / 决策层**\n{business_text}"}
        },
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"🧠 **技术实现层**\n{tech_text}"}
        },
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"🎨 **体验与交付层**\n{delivery_text}"}
        },
        {"tag": "hr"},
        # 行动建议
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"**🚀 行动建议**: {action_label}\n{action_detail}"}
        },
        {"tag": "hr"},
        # 来源
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"**📚 来源**\n{sources_text}"}
        }
    ])

    card_content = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"📰 AI 日报 | {report.get('date', datetime.now().strftime('%Y-%m-%d'))}"},
            "template": "blue"
        },
        "elements": elements
    }

    payload = {
        "timestamp": timestamp,
        "sign": sign,
        "msg_type": "interactive",
        "card": card_content
    }

    # 重试机制
    for attempt in range(3):
        try:
            resp = requests.post(FEISHU_WEBHOOK_URL, json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            if result.get("code") == 0:
                print("[INFO] 飞书推送成功")
                return
            else:
                print(f"[WARN] 飞书推送失败: {result}")
        except Exception as e:
            print(f"[WARN] 飞书推送异常 (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    print("[ERROR] 飞书推送最终失败")


# ==================== 主流程 ====================
def main():
    print(f"[INFO] 开始执行 AI 日报任务 - {datetime.now().isoformat()}")

    # 1. 抓取 RSS
    candidates = fetch_rss_entries()
    if not candidates:
        print("[INFO] 无新内容，退出")
        return

    # 2. 评分
    top_entries = score_entries(candidates)
    if not top_entries:
        print("[INFO] 无高分内容，退出")
        return

    # 3. 生成日报
    report = generate_daily_report(top_entries)
    print("[INFO] 日报生成完成")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # 4. 发送飞书
    send_to_feishu(report)

    # 5. 更新去重文件
    sent_hashes = load_sent_hashes()
    new_hashes = {e["hash"] for e in top_entries}
    sent_hashes.update(new_hashes)
    save_sent_hashes(sent_hashes)
    print(f"[INFO] 已更新去重文件，新增 {len(new_hashes)} 条")


if __name__ == "__main__":
    main()
