import base64
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
import streamlit as st
from bs4 import BeautifulSoup

# =====================================================
# 1. CẤU HÌNH
# =====================================================
st.set_page_config(
    page_title="Laptop Decision", page_icon="💻", layout="wide", initial_sidebar_state="collapsed"
)


# Key KHÔNG viết thẳng vào code. Đặt trong .streamlit/secrets.toml, ví dụ:
#   GROQ_API_KEY = "gsk_xxx"
# hoặc đặt biến môi trường cùng tên.
def load_secret(name: str) -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:  # chưa có secrets.toml hoặc thiếu khóa
        return ""


# Các nhà cung cấp đều dùng chuẩn OpenAI-compatible nên dùng chung 1 hàm gọi.
# Tên model thay đổi theo thời gian: nếu báo lỗi model, hãy xem danh sách model
# mới nhất của nhà cung cấp rồi sửa ở ô Model trên sidebar.
PROVIDERS = {
    "Google Gemini (có gói miễn phí)": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key_name": "GEMINI_API_KEY",
        "model": "gemini-3.8-flash",
    },
    "Groq (có gói miễn phí)": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_name": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "Ollama (chạy trên máy, miễn phí)": {
        "url": "http://localhost:11434/v1/chat/completions",
        "key_name": "",  # không cần key
        "model": "qwen2.5:7b",
    },
    "Hugging Face (cần credits)": {
        "url": "https://router.huggingface.co/v1/chat/completions",
        "key_name": "HF_TOKEN",
        "model": "openai/gpt-oss-120b",
    },
}

MAX_SITES = 6  # link.txt hiện có 6 website
ENOUGH_PRODUCTS = 6  # đủ 4–10 phương án trong ngân sách thì dừng tìm kiếm (đợt đọc 4 trang/lần nên tối đa ~9)
REQUEST_TIMEOUT = 15
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}

SYSTEM_PROMPT = """Bạn là MCee, trợ lý AI tư vấn chọn laptop. Trả lời bằng tiếng Việt, giọng thân thiện, ngắn gọn.

Bạn được cung cấp nội dung các trang đã được thu thập TỪ CÁC WEBSITE do người dùng chỉ định.

QUY TẮC BẮT BUỘC:
1. CHỈ được dùng thông tin nằm trong phần NGUỒN DỮ LIỆU bên dưới. Tuyệt đối không dùng kiến thức bên ngoài, không đoán giá hay cấu hình.
2. Mỗi thông tin (giá, CPU, RAM, ổ cứng, màn hình, pin, hệ điều hành...) phải ghi kèm nguồn dạng [Nguồn n].
3. Nếu nguồn không có thông tin nào đó, ghi "Không có trong nguồn" thay vì tự bịa.
4. Hãy tự lọc các sản phẩm trong nguồn theo ngân sách và nhu cầu của người dùng. "Hãng ưu tiên" chỉ là ưu tiên, KHÔNG phải điều kiện bắt buộc: hãy đưa mẫu của hãng đó lên trước, nhưng nếu có ít hơn 3 mẫu thì bổ sung mẫu hãng khác phù hợp ngân sách và nhu cầu (ghi rõ là hãng khác). Nếu thật sự không có laptop nào phù hợp, nói rõ điều đó.
7. Mỗi mẫu được gợi ý PHẢI kèm link sản phẩm dạng markdown [Tên máy](link), lấy từ dòng "Link:" của đúng nguồn chứa mẫu đó. Không tự tạo link.
5. Cùng một mẫu máy xuất hiện ở nhiều nguồn: so sánh giá giữa các nguồn.
8. Nếu người dùng có "Yêu cầu thêm" (ví dụ RAM 32GB, máy nhẹ, pin trâu...), PHẢI có mục riêng "Đáp ứng yêu cầu thêm" trả lời từng yêu cầu cho các mẫu được gợi ý. Với RAM: mẫu nào đã có sẵn mức đó, mẫu nào hỗ trợ NÂNG CẤP tối đa tới mức đó (theo nguồn, kèm [Nguồn n]), mẫu nào RAM hàn/không có thông tin. Với "nhẹ": nêu trọng lượng nếu nguồn có. Con số người dùng nêu không bắt buộc phải khớp tuyệt đối: máy gần đáp ứng vẫn nên nêu.
9. Người dùng có thể chọn bộ lọc 'Hãng CPU', 'RAM', 'Ổ cứng SSD', 'GPU'. Chỉ gợi ý máy đúng các bộ lọc này (kèm [Nguồn n]); máy mà nguồn không ghi rõ thì nói "Không có trong nguồn" và xếp sau các máy đã xác nhận được. 'Không có GPU rời' nghĩa là máy chỉ dùng card đồ họa tích hợp.
6. Cảnh báo rủi ro nếu thấy dấu hiệu: đánh giá nghe quá tích cực/na ná nhau (nghi review ảo), thông tin sản phẩm thiếu hoặc mâu thuẫn, giá chênh bất thường, người bán không rõ ràng.

ĐỊNH DẠNG TRẢ LỜI:
## Gợi ý phù hợp nhất
Từ 3 đến 5 mẫu (nếu nguồn có đủ), KHÔNG cần lập bảng so sánh (app đã có bảng riêng), mỗi mẫu: tên, giá, hệ điều hành, cấu hình chính, lý do phù hợp với nhu cầu, link sản phẩm (nếu có trong nguồn), nguồn.
## Đáp ứng yêu cầu thêm
Trả lời từng yêu cầu trong mục 'Yêu cầu thêm' (bỏ qua mục này nếu người dùng không có yêu cầu thêm).
## Tóm tắt đánh giá
Tóm tắt review/nhận xét có trong nguồn (ưu, nhược điểm).
## Rủi ro & độ tin cậy nguồn
Nhận xét về độ tin cậy của từng nguồn và các rủi ro phát hiện được.
## Kết luận
Một mẫu nên chọn và vì sao."""


# =====================================================
# 2. TIỆN ÍCH CHUNG
# =====================================================
VND_RE = re.compile(r"(\d{1,3}(?:[.,]\d{3}){1,3})\s*(?:₫|đ|VNĐ|VND)", re.I)
PRICE_RE = re.compile(r"\d{1,3}(?:[.,]\d{3}){1,3}\s*(?:₫|đ|VNĐ|VND)|\d+(?:[.,]\d+)?\s*triệu", re.I)
SKIP_EXT = re.compile(r"\.(jpe?g|png|gif|webp|svg|ico|pdf|zip|rar|mp4|mp3|css|js|xml|gz)$", re.I)
JUNK_TOKEN = re.compile(
    r"^(site-|main-|top-|mobile-)?(menu|footer|popup|modal|cookie|breadcrumbs?|newsletter|social)([-_].*)?$"
)
JUNK_LINK = re.compile(
    r"gio hang|dang nhap|dang ky|tai khoan|chinh sach|lien he|gioi thieu|tuyen dung|bao hanh|"
    r"huong dan|thanh toan|van chuyen|tra gop|dieu khoan|bao mat|sitemap|login|register|cart|"
    r"account|contact|policy|facebook|zalo|youtube|tiktok|checkout"
)


def norm(s: str) -> str:
    """Bỏ dấu tiếng Việt, về chữ thường, chỉ giữ chữ-số: để so khớp từ khóa với URL/văn bản."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def host_of(url: str) -> str:
    h = urlparse(url).netloc.lower()
    return h[4:] if h.startswith("www.") else h


def parse_urls(raw: str) -> list[str]:
    """Tách link/website từ ô nhập (mỗi dòng 1 link), bỏ trùng, giữ thứ tự."""
    urls, seen = [], set()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not re.match(r"^https?://", line):
            line = "https://" + line
        if line not in seen:
            seen.add(line)
            urls.append(line)
    return urls[:MAX_SITES]


LINK_FILE = Path(__file__).parent / "link.txt"  # đặt link.txt cùng thư mục với file app
FALLBACK_SITES = [  # dùng khi không tìm thấy link.txt
    "https://xgear.net/",
    "https://phongvu.vn/",
    "https://www.thegioididong.com/",
    "https://cellphones.com.vn/",
    "https://laptopworld.vn/",
    "https://gearvn.com/",
]


def load_default_sites() -> list[str]:
    """Đọc danh sách website mặc định từ link.txt (mỗi dòng 1 link)."""
    try:
        raw = LINK_FILE.read_text(encoding="utf-8-sig")
    except Exception:
        raw = ""
    return parse_urls(raw) or parse_urls("\n".join(FALLBACK_SITES))


# Lựa chọn lọc theo cấu hình
CPU_BRAND_OPTIONS = ["Không quan tâm", "Intel", "AMD", "Apple", "Qualcomm Snapdragon"]
RAM_OPTIONS = ["Không quan tâm", "8GB", "16GB", "32GB", "Trên 32GB"]
GPU_OPTIONS = ["Không quan tâm", "NVIDIA", "AMD", "Không có GPU rời"]
SSD_OPTIONS = ["Không quan tâm", "256GB", "512GB", "1TB", "2TB"]
_GPU_KIND = {"NVIDIA": "nvidia", "AMD": "amd", "Không có GPU rời": "none"}


def cpu_brand_of(cpu: str) -> str:
    c = (cpu or "").lower()
    if not c:
        return ""
    if "snapdragon" in c:
        return "Qualcomm Snapdragon"
    if re.search(r"apple|\bm[1-9]\b", c):
        return "Apple"
    if re.search(r"ryzen|amd|athlon", c):
        return "AMD"
    if re.search(r"intel|core|celeron|pentium|\bn\d{2,3}\b", c):
        return "Intel"
    return ""


def spec_check(specs: dict, cpu_brand: str, ram_pref: str, gpu_pref: str,
               ssd_pref: str = "Không quan tâm") -> tuple[bool, int]:
    """
    So cấu hình máy với bộ lọc của người dùng.
    Trả về (còn_phù_hợp, số_tiêu_chí_chưa_xác_nhận). Máy chắc chắn SAI bộ lọc -> loại;
    máy mà trang không ghi rõ -> vẫn giữ nhưng xếp sau các máy đã xác nhận đúng.
    """
    unsure = 0
    if cpu_brand != "Không quan tâm":
        b = cpu_brand_of(specs.get("cpu", ""))
        if not b:
            unsure += 1
        elif b != cpu_brand:
            return False, 0
    if ram_pref != "Không quan tâm":
        have, mx = _gb(specs.get("ram", "")), _gb(specs.get("ram_max", ""))
        if not have:
            unsure += 1
        else:
            if ram_pref == "Trên 32GB":
                ok, can_up = have > 32, mx > 32
            else:
                target = int(ram_pref[:-2])
                ok, can_up = have == target, (have < target <= mx)
            if not ok:
                if can_up:
                    unsure += 1  # chưa đúng sẵn nhưng nâng cấp được tới mức này
                else:
                    return False, 0
    if gpu_pref != "Không quan tâm":
        kind = specs.get("gpu_kind", "")
        if not kind:
            unsure += 1
        elif kind != _GPU_KIND[gpu_pref]:
            return False, 0
    if ssd_pref != "Không quan tâm":
        have, target = _gb(specs.get("ssd", "")), _gb(ssd_pref)
        if not have:
            unsure += 1
        elif abs(have - target) > target * 0.05:  # dung sai 5%: 1000GB ≈ 1TB, 500GB ≈ 512GB
            return False, 0
    return True, unsure


def build_keywords(purpose: str, os_pref: str, brand_pref: str,
                   cpu_brand: str = "Không quan tâm", ram_pref: str = "Không quan tâm",
                   gpu_pref: str = "Không quan tâm") -> list[str]:
    """Từ khóa dùng để chọn trang nào đáng đọc trong website."""
    kws = []
    for b in re.split(r"[,;/]+", brand_pref or ""):
        b = norm(b)
        if b:
            kws.append(b)
    purpose_map = {
        "Học tập – văn phòng": ["van phong", "hoc tap", "sinh vien"],
        "Lập trình": ["lap trinh", "van phong"],
        "Đồ họa – thiết kế": ["do hoa", "thiet ke"],
        "Chơi game": ["gaming", "game"],
        "Mỏng nhẹ – di động": ["mong nhe", "ultrabook"],
    }
    kws += purpose_map.get(purpose, [])
    if os_pref == "macOS":
        kws += ["macbook", "apple"]
    kws += {"Intel": ["intel"], "AMD": ["amd", "ryzen"], "Apple": ["macbook", "apple"],
            "Qualcomm Snapdragon": ["snapdragon"]}.get(cpu_brand, [])
    kws += {"NVIDIA": ["rtx", "nvidia", "gaming"], "AMD": ["radeon"]}.get(gpu_pref, [])
    if ram_pref in ("16GB", "32GB"):
        kws.append(norm(ram_pref))
    return kws


# =====================================================
# 3. TẢI TRANG & TRÍCH XUẤT
# =====================================================
def get_html(url: str) -> dict:
    """Tải 1 URL (không dùng streamlit để an toàn khi chạy trong thread)."""
    try:
        r = requests.get(url, headers=BROWSER_HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        if r.encoding is None or r.encoding.lower() == "iso-8859-1":
            r.encoding = r.apparent_encoding
        return {"ok": True, "html": r.text, "error": ""}
    except Exception as e:
        return {"ok": False, "html": "", "error": str(e)}


def extract_json_ld(soup) -> str:
    """Lấy dữ liệu sản phẩm có cấu trúc (JSON-LD) mà nhiều web bán hàng nhúng sẵn."""
    items = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue
        stack = list(data) if isinstance(data, list) else [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
                continue
            if not isinstance(node, dict):
                continue
            if isinstance(node.get("@graph"), list):
                stack.extend(node["@graph"])
            if isinstance(node.get("itemListElement"), list):
                stack.extend(node["itemListElement"])
            if isinstance(node.get("item"), dict):
                stack.append(node["item"])
            t = node.get("@type")
            if t == "Product" or (isinstance(t, list) and "Product" in t):
                offers = node.get("offers", {})
                if isinstance(offers, list):
                    offers = offers[0] if offers else {}
                if not isinstance(offers, dict):
                    offers = {}
                price = offers.get("price") or offers.get("lowPrice") or ""
                cur = offers.get("priceCurrency", "")
                desc = str(node.get("description", ""))[:200].replace("\n", " ")
                items.append(f"- {node.get('name', '')} | giá: {price} {cur} | {desc}")
    return "\n".join(items[:40])


def clean_lines(soup, remove_junk: bool) -> list[str]:
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "svg", "form"]):
        tag.decompose()
    if remove_junk:
        for tag in soup.find_all(True):
            if getattr(tag, "decomposed", False) or tag.name in ("html", "body", "main"):
                continue
            tokens = list(tag.get("class") or []) + ([tag.get("id")] if tag.get("id") else [])
            if any(JUNK_TOKEN.match(t.lower()) for t in tokens):
                tag.decompose()
    raw = [re.sub(r"\s+", " ", ln).strip() for ln in soup.get_text(separator="\n").splitlines()]
    lines, seen = [], set()
    for ln in raw:
        if len(ln) < 2 or ln in seen:
            continue
        seen.add(ln)
        lines.append(ln)
    return lines


def _first(patterns: list[str], text: str, group: int = 0) -> str:
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return re.sub(r"\s+", " ", m.group(group)).strip()
    return ""


def _clean_cpu(v: str) -> str:
    v = re.sub(r"[®™]", "", v)
    v = re.sub(r"(?i)\bcore\b", "Core", v)
    v = re.sub(r"(Core\s*(?:Ultra\s*)?[3579])\s*-\s*(?=\d{3})", r"\1 ", v)  # Core 5-315 -> Core 5 315
    v = re.sub(r"\s+", " ", v).strip(" -/,|")
    if v.startswith("Core"):
        v = "Intel " + v
    return v


def _find_os(blob: str) -> str:
    m = re.search(r"\b(?:Windows|Win)\s*(1[01])(?:\s*(Home|Pro))?(?:\s*(SL|Single\s*Language))?", blob, re.I)
    if m:
        out = f"Windows {m.group(1)}"
        if m.group(2):
            out += " " + m.group(2).title()
        if m.group(3):
            out += " SL"
        return out
    m = re.search(r"FreeDOS|NoOS|No OS|Linux|Ubuntu|macOS|Không\s+(?:kèm\s+)?hệ điều hành", blob, re.I)
    return m.group(0) if m else ""


def _classify_gpu(t: str, generic: bool = True) -> tuple[str, str]:
    """Nhận diện GPU từ 1 đoạn chữ -> (tên, loại: nvidia | amd | none | ''). 'none' = chỉ có card tích hợp."""
    if not t:
        return "", ""
    m = re.search(
        r"\b(?:(?:RTX|GTX)\s?(?:PRO\s*)?A?\d{3,4}(?:\s?Ti)?(?:\s?SUPER)?|MX\s?\d{3}|Quadro\s?\w+)\b", t, re.I
    )
    if m:
        name = re.sub(r"(?i)(RTX|GTX|MX)\s?(?=\S)", lambda x: x.group(1).upper() + " ", m.group(0)).strip()
        return ("NVIDIA " + name if name.lower().startswith("quadro") else "NVIDIA GeForce " + name), "nvidia"
    m = re.search(r"Radeon™?\s*(?:RX|PRO)\s?\w?\d{3,4}\w{0,3}", t, re.I)
    if m:
        return "AMD " + re.sub(r"[®™]", "", m.group(0)).strip(), "amd"
    if re.search(r"NVIDIA\s*GeForce|GeForce", t, re.I):
        return "NVIDIA GeForce", "nvidia"
    integrated = r"Intel®?\s*(?:UHD|Iris|Arc|HD|Graphics)\b|Radeon™?\s*(?:Graphics|\d{3}M)|Adreno|Apple\s*M\d"
    if generic:
        integrated += r"|Onboard|tích hợp|integrated|dùng chung"
    m = re.search(integrated, t, re.I)
    if m:
        return "Tích hợp", "none"
    return "", ""


def extract_specs(title: str, text: str) -> dict:
    """
    Đoán nhanh cấu hình chính từ chữ trên trang (best-effort; thiếu thì để trống).
    Hỗ trợ cách viết phổ biến của web bán lẻ Việt Nam, ví dụ:
      (Core 5 315/ 8GB/ 512GB/ Windows 11)  |  CORE 5-315/8GB/512GB PCIE/14.0 WUXGA/WIN11  |  Core 5 320/AI/16GB/512GB/14"WUXGA/Win11SL
      CPU: Intel® Core™ 3 304 (1.5 GHz...)  |  Công nghệ CPU | Intel Core™ 5 320
    """
    body = text.split("### Toàn bộ nội dung trang")[-1][:15000]  # bỏ phần 'đoạn có giá' (hay lẫn máy khác)
    blob = title + "\n" + body

    cpu = _first([
        r"(?:Intel®?\s*)?Core™?\s*(?:Ultra\s*)?[3579]\s*[- ]?\s*\d{3}\w?\b",
        r"(?:Intel®?\s*)?Core™?\s*i[3579][- ]?\s?\d{4,5}\w{0,2}",
        r"(?:AMD\s*)?Ryzen™?\s*(?:AI\s*)?(?:Z\d\s*)?[3579]\s*\d{3,5}\w{0,3}",
        r"Apple\s*M\d(?:\s*(?:Pro|Max|Ultra))?",
        r"Snapdragon™?\s*X\w*(?:\s*(?:Plus|Elite))?",
        r"Intel®?\s*(?:Processor\s*)?N\d{2,3}\b",
        r"(?:Celeron|Pentium)®?\s*\w{1,8}",
    ], blob)
    if cpu:
        cpu = _clean_cpu(cpu)
    else:  # dự phòng: lấy giá trị đứng sau nhãn "CPU" / "Bộ vi xử lý"
        lab = _first([r"(?:CPU|Bộ vi xử lý|Vi xử lý|Bộ xử lý)\s*[:|\n]\s*([^\n|]{3,60})"], blob, 1)
        cpu = re.sub(r"\s*\(.*$", "", re.sub(r"[®™]", "", lab)).strip() if lab else ""

    ram = ""
    for pat in (
        r"(?:Dung lượng RAM|RAM|Bộ nhớ trong)[\s:|]*(\d{1,2})\s?GB",
        r"(\d{1,2})\s?GB\s*(?:LP)?DDR\d",
        r"(\d{1,2})\s?GB\s*RAM",
        r"[/(,]\s*(\d{1,2})\s?GB\s*[/,)]",  # trong tên máy: (Core 5 315/ 8GB/ 512GB/ ...)
    ):
        m = re.search(pat, blob, re.I)
        if m and int(m.group(1)) <= 64:
            ram = m.group(1)
            break

    ssd = _first([
        r"(\d{3,4}\s?GB)\s*(?:PCIe|NVMe|SSD|M\.2)",
        r"(\d\s?TB)\s*(?:PCIe|NVMe|SSD|M\.2)",
        r"(?:SSD|Ổ cứng|Dung lượng ổ cứng)[\s:|]*(?:SSD\s*)?(\d{3,4}\s?GB|\d\s?TB)",
        r"/\s*(\d{3,4}\s?GB|\d\s?TB)\s*[/,)]",  # trong tên máy
    ], blob, 1)

    screen = _first([
        r"\b(1[1-8](?:[.,]\d)?)\s?(?:inch|in\b|\"|”|″|'')",
        r"Màn hình[\s:|]*(1[1-8](?:[.,]\d)?)",
        r"/\s*(1[1-8](?:[.,]\d)?)\s*(?:\"|”|″)?\s*(?:WUXGA|FHD|QHD|WQXGA|WQHD|OLED|IPS|2K|3K|4K|HD)",
    ], blob, 1)
    if screen:
        screen = screen.replace(",", ".")
        screen = screen[:-2] if screen.endswith(".0") else screen

    # RAM tối đa nâng cấp được (nếu trang có ghi) và trọng lượng
    ram_max = ""
    for pat in (
        r"RAM[^\n]{0,90}?(?:nâng cấp|nâng|tối đa|up to|max(?:imum)?|expandable)[^\n]{0,30}?(\d{2,3})\s?GB",
        r"(?:nâng cấp|tối đa|up to|max(?:imum)?)[^\n]{0,40}?RAM[^\n]{0,30}?(\d{2,3})\s?GB",
        r"(?:Số khe RAM|Khe RAM|Slot RAM)[^\n]{0,80}?(?:tối đa|up to|max)[^\n]{0,20}?(\d{2,3})\s?GB",
    ):
        m = re.search(pat, blob, re.I)
        if m and int(m.group(1)) in (16, 24, 32, 48, 64, 96, 128):
            ram_max = m.group(1)
            break
    weight = _first([
        r"(?:Trọng lượng|Cân nặng|Khối lượng|Weight)[\s:|~≈]*(\d(?:[.,]\d{1,2})?)\s?kg",
        r"\b(\d[.,]\d{1,2})\s?kg\b",
    ], blob, 1).replace(",", ".")
    try:
        weight = weight if 0.7 <= float(weight) <= 4.5 else ""
    except ValueError:
        weight = ""

    # GPU: ưu tiên tên máy, rồi dòng có nhãn (Card đồ họa/VGA/GPU), cuối cùng mới quét cả trang
    gpu, gpu_kind = _classify_gpu(title)
    if not gpu_kind:
        lab = _first([r"(?:Card đồ họa|Card màn hình|Card rời|Đồ họa|VGA|GPU)\s*[:|\n]\s*([^\n]{3,100})"], blob, 1)
        gpu, gpu_kind = _classify_gpu(lab)
    if not gpu_kind:
        gpu, gpu_kind = _classify_gpu(blob, generic=False)

    return {
        "cpu": cpu,
        "gpu": gpu,
        "gpu_kind": gpu_kind,
        "ram_max": f"{ram_max}GB" if ram_max else "",
        "weight": f"{weight}kg" if weight else "",
        "ram": f"{ram}GB" if ram else "",
        "ssd": ssd.replace(" ", "") if ssd else "",
        "screen": f'{screen}"' if screen else "",
        "os": _find_os(blob),
    }


def looks_like_laptop(url: str, title: str, specs: dict) -> bool:
    """Loại các sản phẩm không phải laptop (PC để bàn, phụ kiện...)."""
    path = urlparse(url).path
    if re.search(r"laptop|macbook|notebook", norm(title + " " + path)):
        return True
    return bool(specs.get("screen"))  # có kích thước màn hình 11–18 inch


def extract_page(url: str, html: str) -> dict:
    """Trích xuất nội dung hữu ích của 1 trang: dữ liệu sản phẩm, các đoạn có giá, rồi toàn bộ chữ."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else url
    structured = extract_json_ld(soup)  # lấy trước khi xóa thẻ script

    lines = clean_lines(soup, remove_junk=True)
    if sum(len(x) for x in lines) < 300:  # lỡ xóa nhầm quá tay -> đọc lại không lọc
        lines = clean_lines(BeautifulSoup(html, "html.parser"), remove_junk=False)

    # Gom các đoạn có giá (kèm 2 dòng phía trên: thường là tên + thông số) lên đầu
    price_blocks, used = [], set()
    for i, ln in enumerate(lines):
        if PRICE_RE.search(ln):
            for j in range(max(0, i - 2), i + 1):
                if j not in used:
                    used.add(j)
                    price_blocks.append(lines[j])

    parts = []
    if structured:
        parts.append("### Dữ liệu sản phẩm có cấu trúc\n" + structured)
    if price_blocks:
        parts.append("### Các đoạn có giá trên trang\n" + "\n".join(price_blocks[:150]))
    parts.append("### Toàn bộ nội dung trang\n" + "\n".join(lines))
    text = "\n\n".join(parts)

    # Giá (VNĐ) tìm thấy trên trang: ưu tiên dữ liệu có cấu trúc, rồi tới giá đầu tiên trong nội dung
    prices = []
    for m in re.finditer(r"giá: (\d{6,9})", structured):
        prices.append(int(m.group(1)))
    for m in VND_RE.finditer("\n".join(price_blocks)):
        prices.append(int(re.sub(r"\D", "", m.group(1))))
    prices = [p for p in prices if 1_000_000 <= p <= 300_000_000]

    # Chữ ký nội dung: dùng để loại các trang trùng nhau (VD: cùng danh mục nhưng khác tham số sắp xếp/lọc)
    sig_source = "\n".join(price_blocks[:60]) if price_blocks else text[:800]
    specs = extract_specs(title, text)
    return {
        "url": url,
        "title": title,
        "text": text,
        "ok": len(text) >= 200,
        "has_price": bool(price_blocks) or bool(structured),
        "sig": hash(sig_source),
        "price": prices[0] if prices else None,
        "is_product": False,  # được đánh dấu lại trong crawl_site
        "specs": specs,
        "is_laptop": looks_like_laptop(url, title, specs),
    }


def extract_links(base_url: str, html: str) -> list[dict]:
    """Lấy các link NỘI BỘ cùng website. Đánh dấu link nào nằm trong 'thẻ sản phẩm' (gần giá tiền)."""
    soup = BeautifulSoup(html, "html.parser")
    base_host = host_of(base_url)
    out = {}
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"]).split("#")[0]
        p = urlparse(href)
        if p.scheme not in ("http", "https") or host_of(href) != base_host or SKIP_EXT.search(p.path):
            continue
        text = a.get_text(" ", strip=True)
        card, price_val, node = False, None, a
        for _ in range(4):
            node = node.parent
            if node is None:
                break
            t = node.get_text(" ", strip=True)
            if len(t) > 700:
                break
            if PRICE_RE.search(t):
                card = True
                m = VND_RE.search(t)
                if m:
                    price_val = int(re.sub(r"\D", "", m.group(1)))
                break
        rec = out.setdefault(href, {"url": href, "text": text, "card": card, "price": price_val})
        if len(text) > len(rec["text"]):
            rec["text"] = text
        rec["card"] = rec["card"] or card
        if price_val and not rec.get("price"):
            rec["price"] = price_val
    return list(out.values())


def score_link(link: dict, keywords: list[str], lo: int = 0, hi: int = 0) -> int:
    """Chấm điểm 1 link: càng liên quan laptop/nhu cầu người dùng càng cao."""
    p = urlparse(link["url"])
    blob = norm(p.path + " " + p.query + " " + link["text"])
    score = 0
    for k in keywords:
        if k and k in blob:
            score += 3
    if re.search(r"\blaptop\b|may tinh xach tay|notebook|macbook", blob):
        score += 2
    if re.search(r"danh gia|review|so sanh", blob):
        score += 1
    if link.get("card"):
        score += 4  # nhiều khả năng là sản phẩm
    price = link.get("price")
    if price and lo and hi:  # ưu tiên sản phẩm có giá nằm trong ngân sách của người dùng
        if lo <= price <= hi:
            score += 6
        elif lo * 0.85 <= price <= hi * 1.15:
            score += 1
        else:
            score -= 6
    if re.search(r"\bpc\b|pchm|de ban|all in one|linh kien|phu kien|chuot|ban phim|tai nghe|\bman hinh\b|may in|camera|\bloa\b", blob):
        score -= 4  # không phải laptop
    if p.query:
        score -= 3  # link có tham số lọc/sắp xếp thường là bản trùng của trang khác
    if JUNK_LINK.search(blob):
        score -= 10
    return score


def sitemap_urls(origin: str, robots_text: str, limit: int = 3000) -> list[str]:
    """Phương án dự phòng: lấy danh sách URL từ sitemap khi trang tải sản phẩm bằng JavaScript."""
    maps = re.findall(r"(?im)^sitemap:\s*(\S+)", robots_text) or [origin + "/sitemap.xml"]
    urls, queue, fetched = [], list(maps[:3]), 0
    while queue and fetched < 6 and len(urls) < limit:
        queue.sort(key=lambda u: 0 if re.search(r"product|san-?pham|laptop", u, re.I) else 1)
        sm = queue.pop(0)
        fetched += 1
        d = get_html(sm)
        if not d["ok"]:
            continue
        for loc in re.findall(r"<loc>\s*(.*?)\s*</loc>", d["html"], re.S):
            if loc.endswith(".xml"):
                queue.append(loc)
            elif not loc.endswith(".gz"):
                urls.append(loc)
    return urls[:limit]


@st.cache_data(show_spinner=False, ttl=60 * 30)
def crawl_site(start_url: str, keywords: tuple, max_pages: int, lo: int, hi: int) -> tuple[list[dict], list[str]]:
    """
    Khám phá 1 website, CHỈ trong cùng tên miền với link người dùng đưa:
      Bước 0: đọc trang bắt đầu
      Bước 1: vào các trang danh mục/liên quan nhất (chấm điểm theo từ khóa)
      Bước 2: vào các trang sản phẩm tìm được từ danh mục
      Dự phòng: nếu ít trang có giá thì thử sitemap
    Trả về (danh sách trang, nhật ký).
    """
    kws = list(keywords)

    def sc(link: dict) -> int:
        return score_link(link, kws, lo, hi)

    logs: list[str] = []
    origin = f"{urlparse(start_url).scheme}://{urlparse(start_url).netloc}"

    robots_text = get_html(origin + "/robots.txt")["html"]
    rp = RobotFileParser()
    rp.parse(robots_text.splitlines())

    def allowed(u: str) -> bool:
        try:
            return rp.can_fetch("*", u)
        except Exception:
            return True

    visited: set[str] = set()
    pages: list[dict] = []
    pool: dict[str, dict] = {}
    seen_sigs: set[int] = set()

    def fetch_many(urls: list[str]):
        urls = [u for u in urls if u not in visited]
        for u in urls:
            visited.add(u)
        blocked = [u for u in urls if not allowed(u)]
        for u in blocked:
            logs.append(f"⛔ Bỏ qua (robots.txt không cho phép): {u}")
        urls = [u for u in urls if u not in blocked]
        if not urls:
            return
        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(get_html, urls))
        for u, d in zip(urls, results):
            if not d["ok"]:
                logs.append(f"⚠️ Không tải được {u}: {d['error']}")
                continue
            page = extract_page(u, d["html"])
            page["is_product"] = bool(pool.get(u, {}).get("card"))  # được liên kết từ 'thẻ sản phẩm'
            page["title"] = page["title"].split(" | ")[0].strip() or page["title"]
            for l in extract_links(u, d["html"]):
                cur = pool.get(l["url"])
                if cur is None:
                    pool[l["url"]] = l
                else:
                    cur["card"] = cur["card"] or l["card"]
                    if l.get("price") and not cur.get("price"):
                        cur["price"] = l["price"]
                    if len(l["text"]) > len(cur["text"]):
                        cur["text"] = l["text"]
            if page["ok"]:
                same_title = sum(1 for p in pages if p["title"] == page["title"])
                if page["sig"] in seen_sigs or same_title >= 2:
                    logs.append(f"♻️ Bỏ qua trang trùng nội dung: {u}")
                    continue
                seen_sigs.add(page["sig"])
                pages.append(page)
                logs.append(f"📄 Đã đọc: {page['title']} — {u}")
            else:
                logs.append(f"⚠️ Trang quá ít nội dung (có thể tải bằng JavaScript): {u}")

    def pick(n: int) -> list[str]:
        ranked = sorted(pool.values(), key=lambda l: -sc(l))
        return [l["url"] for l in ranked if l["url"] not in visited and sc(l) > 0][:n]

    def good_count() -> int:
        return sum(
            1 for p in pages
            if p.get("is_product") and p.get("is_laptop") and p.get("price") and lo <= p["price"] <= hi
        )

    # Bước 0
    fetch_many([start_url])
    # Bước 1: danh mục liên quan
    fetch_many(pick(min(max(2, max_pages // 3), max_pages - len(visited))))
    # Bước 2: sản phẩm, đọc từng đợt nhỏ và DỪNG SỚM khi đã đủ phương án để giảm tải
    while good_count() < ENOUGH_PRODUCTS and len(visited) < max_pages:
        batch = pick(min(4, max_pages - len(visited)))
        if not batch:
            break
        fetch_many(batch)
    logs.append(f"✅ Tìm được {good_count()} phương án trong ngân sách ({len(pages)} trang đã đọc)")

    # Dự phòng: sitemap
    remaining = max_pages - len(visited)
    if sum(p["has_price"] for p in pages) < 3 and remaining > 0:
        logs.append("🗺️ Ít trang có giá, thử đọc sitemap của website...")
        cands = [
            {"url": u, "text": "", "card": False}
            for u in sitemap_urls(origin, robots_text)
            if host_of(u) == host_of(start_url) and u not in visited
        ]
        ranked = sorted(cands, key=lambda l: -sc(l))
        fetch_many([l["url"] for l in ranked if sc(l) > 0][:remaining])

    return pages, logs


# =====================================================
# 4. GỌI MODEL
# =====================================================
def build_context(sources: list[dict]) -> str:
    parts = []
    for i, s in enumerate(sources, start=1):
        parts.append(f"[Nguồn {i}] {s['title']}\nLink: {s['url']}\nNội dung:\n{s['text']}")
    return "\n\n-----\n\n".join(parts)


def ask_model(api_url: str, api_key: str, model: str, criteria: str, context: str, timeout: int = 120) -> str:
    user_msg = (
        f"NHU CẦU / TIÊU CHÍ CỦA NGƯỜI DÙNG:\n{criteria}\n\n"
        f"NGUỒN DỮ LIỆU (chỉ được dùng thông tin trong này):\n{context}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 8000,  # model suy luận tính cả phần "nghĩ" vào giới hạn này nên cần đủ lớn
        "temperature": 0.2,  # thấp để bám sát nguồn, hạn chế bịa
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Mã lỗi {resp.status_code}: {resp.text[:500]}")
    choice = resp.json()["choices"][0]
    text = choice["message"].get("content") or ""
    if choice.get("finish_reason") == "length":
        if not text.strip():
            raise RuntimeError("AI dùng hết giới hạn token trước khi viết xong")
        text += "\n\n---\n⚠️ *Câu trả lời bị cắt do hết giới hạn token. Hãy giảm số trang/ký tự hoặc đổi model rồi thử lại.*"
    return text


def ask_with_fallback(chain: list[dict], criteria: str, sources: list[dict]) -> tuple[str, str]:
    """
    Gọi AI theo danh sách nhà cung cấp `chain` (phần tử đầu là nhà cung cấp đang chọn, các phần tử sau là
    dự phòng nếu có key trong secrets). Với nhà cung cấp chính: thử tối đa 3 lần với dữ liệu giảm dần
    (100% -> 60% -> 35%); nhà cung cấp dự phòng: thử 1 lần với dữ liệu gọn. Lỗi luôn in ra terminal.
    """
    last_error = ""
    for idx, prov in enumerate(chain):
        plan = ((1.0, 100), (0.6, 80), (0.35, 60)) if idx == 0 else ((0.3, 70),)
        for factor, timeout in plan:
            trimmed = [{**s, "text": s["text"][: max(800, int(len(s["text"]) * factor))]} for s in sources]
            try:
                text = ask_model(prov["url"], prov["key"], prov["model"], criteria, build_context(trimmed), timeout=timeout)
                if text and text.strip():
                    return text, ""
                last_error = "AI trả về nội dung rỗng"
            except Exception as e:
                last_error = str(e)
            print(f"[MCee] {prov['model']} thất bại (dữ liệu {int(factor * 100)}%): {last_error}", file=sys.stderr)
            if re.search(r"Mã lỗi (400|401|403|404)", last_error):
                break  # lỗi cấu hình (key/model sai), thử lại cũng vô ích -> sang nhà cung cấp khác
            limited = re.search(r"429|quota|RESOURCE_EXHAUSTED|rate", last_error, re.I)
            time.sleep(15 if limited else 3)  # bị giới hạn lượt gọi thì chờ lâu hơn
    return "", last_error


def _cpu_rank(cpu: str) -> int:
    c = (cpu or "").lower()
    m = re.search(r"core\s*(?:ultra\s*)?([3579])\b", c) or re.search(r"\bi([3579])[- ]", c) or re.search(r"ryzen\s*(?:ai\s*)?([3579])", c)
    rank = int(m.group(1)) if m else 0
    if "ultra" in c:
        rank += 1
    if re.search(r"apple\s*m\d", c):
        rank += 4
    return rank


def _gb(x: str) -> int:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(TB|GB)", x or "", re.I)
    if not m:
        return 0
    v = float(m.group(1))
    return int(v * 1024) if m.group(2).upper() == "TB" else int(v)


def quick_review(products: list[dict], purpose: str, brand_pref: str, extra: str = "") -> str:
    """Nhận xét nhanh dựng tự động từ bảng so sánh: dùng khi phần phân tích chi tiết của AI chưa sẵn sàng."""
    def price_txt(s):
        return f"{s['price']:,}".replace(",", ".") + " ₫"

    def link(s):
        return f"[{s['title']}]({s['url']})"

    def power(s):
        sp = s.get("specs") or {}
        return (_cpu_rank(sp.get("cpu", "")), _gb(sp.get("ram", "")), _gb(sp.get("ssd", "")))

    lines = []
    cheapest = min(products, key=lambda s: s["price"])
    strongest = max(products, key=power)
    if strongest is cheapest and power(strongest) != (0, 0, 0) and len(products) > 1:
        lines.append(f"- 💎 **Vừa rẻ vừa mạnh nhất:** {link(cheapest)} — {price_txt(cheapest)}")
    else:
        lines.append(f"- 💰 **Giá tốt nhất:** {link(cheapest)} — {price_txt(cheapest)}")
        if power(strongest) != (0, 0, 0) and strongest is not cheapest:
            sp = strongest.get("specs") or {}
            detail = ", ".join(x for x in (sp.get("cpu"), sp.get("ram"), sp.get("ssd")) if x)
            lines.append(f"- ⚡ **Cấu hình mạnh nhất:** {link(strongest)} ({detail}) — {price_txt(strongest)}")

    def inch(s):
        m = re.search(r"\d+(?:\.\d+)?", (s.get("specs") or {}).get("screen", ""))
        return float(m.group(0)) if m else None

    sized = [(inch(s), s) for s in products if inch(s)]
    if len({i for i, _ in sized}) > 1:
        small = min(sized, key=lambda t: t[0])[1]
        lines.append(f"- 🪶 **Gọn nhẹ nhất (màn hình nhỏ nhất):** {link(small)} ({(small.get('specs') or {}).get('screen')})")

    brands = [norm(b) for b in re.split(r"[,;/]+", brand_pref or "") if norm(b)]
    liked = [s for s in products if any(b in norm(s["title"]) for b in brands)]
    if brands and liked:
        lines.append(f"- 🏷️ **Đúng hãng bạn ưu tiên:** có {len(liked)} máy trong bảng.")

    # --- đáp ứng 'Yêu cầu thêm' của người dùng (RAM mong muốn, máy nhẹ) ---
    def short(s):
        return re.sub(r"^Laptop\s+", "", s["title"])

    m = re.search(r"(\d{1,3})\s?gb", extra or "", re.I)
    if m and re.search(r"ram|bộ nhớ", extra or "", re.I) or (m and int(m.group(1)) in (16, 24, 32, 48, 64)):
        target = int(m.group(1))
        now, upg, unk = [], [], []
        for s in products:
            sp = s.get("specs") or {}
            have, mx = _gb(sp.get("ram", "")), _gb(sp.get("ram_max", ""))
            if have >= target:
                now.append(s)
            elif mx >= target:
                upg.append(s)
            else:
                unk.append(s)
        parts = []
        if now:
            parts.append("đã có sẵn: " + "; ".join(f"[{short(s)}]({s['url']})" for s in now[:3]))
        if upg:
            parts.append(f"nâng cấp được tới {target}GB (theo trang): " + "; ".join(
                f"[{short(s)}]({s['url']}) (tối đa {s['specs']['ram_max']})" for s in upg[:3]))
        if unk:
            parts.append(f"{len(unk)} máy chưa thấy thông tin nâng cấp RAM trên trang, nên hỏi lại cửa hàng "
                         "(một số dòng RAM hàn trên bo mạch, không nâng được)")
        lines.append(f"- 🧠 **Yêu cầu RAM {target}GB** (không nhất thiết đúng con số này): " + " — ".join(parts) + ".")

    if re.search(r"\bnhẹ\b|mỏng", extra or "", re.I):
        weighed = [(float(re.search(r"\d+(?:\.\d+)?", (s.get("specs") or {}).get("weight", "")).group(0)), s)
                   for s in products if re.search(r"\d", (s.get("specs") or {}).get("weight", ""))]
        if weighed:
            w, s = min(weighed, key=lambda t: t[0])
            lines.append(f"- 🪶 **Yêu cầu máy nhẹ:** nhẹ nhất theo trang là [{short(s)}]({s['url']}) ({w:g}kg).")
        else:
            lines.append("- 🪶 **Yêu cầu máy nhẹ:** các trang chưa ghi rõ cân nặng, bạn có thể tham khảo kích thước màn hình ở bảng.")

    tips = {
        "Lập trình": "nên ưu tiên RAM từ 16GB và ổ SSD 512GB trở lên",
        "Đồ họa – thiết kế": "nên ưu tiên CPU mạnh, RAM từ 16GB và màn hình đẹp",
        "Chơi game": "nên ưu tiên card đồ họa rời, hãy kiểm tra kỹ trong trang sản phẩm",
        "Học tập – văn phòng": "RAM 8GB và SSD 512GB là đủ dùng",
        "Mỏng nhẹ – di động": "nên ưu tiên màn hình 13–14 inch và pin lâu",
    }
    if purpose in tips:
        lines.append(f"- 🎯 **Gợi ý chung cho nhu cầu {purpose.lower()}:** {tips[purpose]}.")
    return "\n".join(lines)


# =====================================================
# 4b. HIỆU ỨNG CHỜ (MCee) & BẢNG SO SÁNH
# =====================================================
AI_NAME = "MCee"
MASCOT_DIR = Path(__file__).parent / "assets"  # đặt ảnh vào đây, đặt tên: stitch.png (hoặc .gif/.webp/.jpg)


def mascot_data_uri() -> str:
    """Đọc ảnh mascot trong thư mục assets và nhúng vào trang dưới dạng data URI."""
    mimes = {"png": "image/png", "gif": "image/gif", "webp": "image/webp", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
    for ext, mime in mimes.items():
        path = MASCOT_DIR / f"stitch.{ext}"
        if path.exists():
            try:
                return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
            except Exception:
                return ""
    return ""


# Các dòng lệnh của vài thuật toán kinh điển; mỗi lần hiển thị sẽ được xáo trộn ngẫu nhiên rồi bay lên
ALGO_LINES = [
    # QuickSort
    "int p = a[(l + r) / 2];", "while (a[i] < p) i++;", "swap(a[i], a[j]);", "quickSort(a, l, j);",
    # Binary search
    "int mid = lo + (hi - lo) / 2;", "if (a[mid] == x) return mid;", "else if (a[mid] < x) lo = mid + 1;",
    "else hi = mid - 1;",
    # Dijkstra
    "pq.push({0, src});", "dist.assign(n, INF);", "if (d > dist[u]) continue;",
    "for (auto [v, w] : adj[u])", "dist[v] = dist[u] + w;",
    # BFS / DFS
    "q.push(start);", "while (!q.empty()) {", "int u = q.front(); q.pop();", "visited[v] = true;",
    "void dfs(int u) {", "if (!vis[v]) dfs(v);",
    # Merge sort
    "int m = (l + r) >> 1;", "mergeSort(a, l, m);", "merge(a, l, m, r);",
    # Euclid / Sàng nguyên tố
    "return b ? gcd(b, a % b) : a;", "for (j = i * i; j <= n; j += i)", "isPrime[j] = false;",
    # Quy hoạch động: Fibonacci, Knapsack, Kadane
    "dp[n] = dp[n-1] + dp[n-2];", "dp[i][w] = max(dp[i-1][w],", "dp[i-1][w-wt[i]] + val[i]);",
    "cur = max(a[i], cur + a[i]);", "best = max(best, cur);",
    # Union-Find, KMP
    "parent[find(a)] = find(b);", "while (k > 0 && s[i] != p[k])", "k = pi[k - 1];",
]
BRAND_TOKENS = ["FITHCMUS", "// FITHCMUS"]  # luôn có trong số các dòng bay lên


def _code_rain_css() -> tuple[str, str]:
    """Chọn ngẫu nhiên và xáo trộn các dòng thuật toán; vị trí/tốc độ bay cố định theo seed cho ổn định."""
    layout = random.Random(2026)
    tokens = random.sample(ALGO_LINES, 20) + BRAND_TOKENS
    random.shuffle(tokens)
    spans, rules = [], []
    for i, tok in enumerate(tokens, start=1):
        safe = tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        spans.append(f'<span class="mc-code">{safe}</span>')
        rules.append(
            f".mc-code:nth-child({i}){{left:{layout.randint(2, max(3, 86 - int(len(tok) * 1.2)))}%;font-size:{layout.randint(12, 19)}px;"
            f"animation-duration:{layout.randint(14, 28)}s;animation-delay:-{layout.randint(0, 28)}s;}}"
        )
    return "".join(spans), "".join(rules)


_CONFETTI_COLORS = ["#7dd3fc", "#ffffff", "#facc15", "#38bdf8", "#f472b6", "#a5f3fc", "#60a5fa"]
_CONFETTI_N = 30


def _streams_css() -> str:
    rng = random.Random(7)
    return "".join(
        f".mc-streams b:nth-child({i}){{left:{rng.randint(2, 97)}%;animation-duration:{rng.randint(3, 8)}s;"
        f"animation-delay:-{rng.randint(0, 8)}s;opacity:.{rng.randint(4, 9)};}}"
        for i in range(1, 19)
    )


def _confetti_css() -> str:
    rng = random.Random(99)
    rules = []
    for i in range(1, _CONFETTI_N + 1):
        ang = rng.uniform(0, 6.2832)
        dist = rng.randint(220, 520)
        dx, dy = int(dist * math.cos(ang)), int(dist * math.sin(ang)) - 40
        rules.append(
            f".mc-confetti i:nth-child({i}){{--dx:{dx}px;--dy:{dy}px;--rot:{rng.randint(-540, 540)}deg;"
            f"background:{_CONFETTI_COLORS[i % len(_CONFETTI_COLORS)]};animation-delay:{rng.uniform(0.05, 0.3):.2f}s;}}"
        )
    return "".join(rules)


def loader_html(text: str) -> str:
    """
    Lớp phủ toàn màn hình (modal) chồng lên trang hiện tại trong lúc MCee làm việc.
    Nền phong cách "hacker/IT": gradient xanh HCMUS, lưới chuyển động, dòng code bay lên, tia quét.
    Thẻ giữa luôn nền sáng và ảnh dùng mix-blend-mode: multiply nên phần nền trắng của ảnh
    sẽ "tan" vào nền thẻ, không còn khung trắng vuông quanh nhân vật.
    """
    uri = mascot_data_uri()
    img = f'<img class="mc-img" src="{uri}" alt="{AI_NAME}">' if uri else '<div class="mc-img mc-fallback">🤖</div>'
    code_spans, code_rules = _code_rain_css()
    css = (
        # --- ẩn thanh Deploy / nút sidebar của Streamlit trong lúc loading (quy tắc mất cùng lớp phủ) ---
        "header[data-testid='stHeader'],[data-testid='stToolbar'],[data-testid='stSidebarCollapsedControl'],"
        "[data-testid='stExpandSidebarButton'],[data-testid='collapsedControl']{opacity:0!important;pointer-events:none!important;}"
        # --- nền ---
        ".mc-overlay{position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:999999;overflow:hidden;"
        "display:flex;align-items:center;justify-content:center;animation:mc-fade .3s ease-out;"
        "background:radial-gradient(60% 55% at 18% 12%,rgba(0,163,255,.40),transparent 62%),"
        "radial-gradient(55% 55% at 88% 90%,rgba(56,189,248,.30),transparent 62%),"
        "radial-gradient(40% 40% at 70% 15%,rgba(250,204,21,.10),transparent 65%),"
        "linear-gradient(135deg,#040a1c 0%,#08265f 52%,#0a4aa6 100%);}"
        ".mc-grid{position:absolute;top:0;left:0;width:100%;height:100%;"
        "background-image:linear-gradient(rgba(125,211,252,.13) 1px,transparent 1px),"
        "linear-gradient(90deg,rgba(125,211,252,.13) 1px,transparent 1px);background-size:56px 56px;"
        "-webkit-mask-image:radial-gradient(circle at center,#000 0%,transparent 78%);"
        "mask-image:radial-gradient(circle at center,#000 0%,transparent 78%);"
        "animation:mc-grid 9s linear infinite;}"
        "@keyframes mc-grid{to{background-position:56px 56px,56px 56px}}"
        ".mc-scan{position:absolute;top:0;left:0;width:100%;height:140px;"
        "background:linear-gradient(transparent,rgba(56,189,248,.16),transparent);"
        "animation:mc-scan 5.5s linear infinite;}"
        "@keyframes mc-scan{0%{transform:translateY(-160px)}100%{transform:translateY(100vh)}}"
        ".mc-vignette{position:absolute;top:0;left:0;width:100%;height:100%;"
        "background:radial-gradient(circle at center,transparent 45%,rgba(2,6,23,.65) 100%);}"
        # --- dòng code bay lên ---
        ".mc-rain{position:absolute;top:0;left:0;width:100%;height:100%;}"
        ".mc-code{position:absolute;bottom:-8%;white-space:nowrap;font-family:Consolas,'Courier New',monospace;"
        "color:rgba(125,211,252,.55);text-shadow:0 0 10px rgba(56,189,248,.8);opacity:0;"
        "animation:mc-rise linear infinite;}"
        ".mc-code:nth-child(3n){color:rgba(165,243,252,.5);}"
        ".mc-code:nth-child(5n){color:rgba(250,204,21,.45);text-shadow:0 0 10px rgba(250,204,21,.6);}"
        "@keyframes mc-rise{0%{transform:translateY(0);opacity:0}12%{opacity:.8}85%{opacity:.5}"
        "100%{transform:translateY(-115vh);opacity:0}}"
        + code_rules +
        # --- góc thương hiệu ---
        ".mc-brand{position:absolute;top:22px;left:28px;z-index:2;font-family:Consolas,'Courier New',monospace;"
        "color:#bae6fd;letter-spacing:.14em;font-size:.95rem;text-shadow:0 0 12px rgba(56,189,248,.9);}"
        ".mc-foot{position:absolute;bottom:22px;left:0;width:100%;text-align:center;z-index:2;"
        "font-family:Consolas,'Courier New',monospace;color:rgba(186,230,253,.75);letter-spacing:.2em;"
        "font-size:.72rem;}"
        # --- thẻ giữa với viền gradient chạy ---
        ".mc-ring{position:relative;z-index:3;border-radius:26px;padding:3px;max-width:90vw;"
        "background:linear-gradient(120deg,#38bdf8,#2563eb,#facc15,#a5f3fc,#1d4ed8,#38bdf8);"
        "background-size:300% 300%;animation:mc-ring 4s linear infinite,mc-glow 2.4s ease-in-out infinite,"
        "mc-enter .75s cubic-bezier(.2,1.5,.4,1) both;}"
        "@keyframes mc-enter{0%{transform:scale(.55);opacity:0}100%{transform:scale(1);opacity:1}}"
        "@keyframes mc-ring{0%{background-position:0% 50%}100%{background-position:300% 50%}}"
        "@keyframes mc-glow{0%,100%{box-shadow:0 0 40px rgba(56,189,248,.40),0 24px 60px rgba(0,0,0,.55)}"
        "50%{box-shadow:0 0 85px rgba(56,189,248,.75),0 24px 60px rgba(0,0,0,.55)}}"
        ".mc-card{isolation:isolate;background:#ffffff;color:#1f2937;border-radius:23px;padding:26px 46px 22px;"
        "display:flex;flex-direction:column;align-items:center;gap:8px;}"
        ".mc-img{height:150px;width:auto;max-width:70vw;mix-blend-mode:multiply;"
        "animation:mc-float 1.1s ease-in-out infinite;}"
        ".mc-fallback{font-size:92px;line-height:150px;height:150px;}"
        ".mc-runner{position:relative;height:44px;width:60px;font-size:34px;line-height:44px;text-align:center;}"
        ".mc-runner span{position:absolute;left:0;right:0;top:0;}"
        ".mc-a{animation:mc-swap-a 1s steps(1) infinite;}"
        ".mc-b{opacity:0;animation:mc-swap-b 1s steps(1) infinite;}"
        "@keyframes mc-fade{0%{opacity:0}100%{opacity:1}}"
        "@keyframes mc-swap-a{0%{opacity:1}50%{opacity:0}100%{opacity:1}}"
        "@keyframes mc-swap-b{0%{opacity:0}50%{opacity:1}100%{opacity:0}}"
        "@keyframes mc-float{0%,100%{transform:translateY(0)}50%{transform:translateY(-9px)}}"
        ".mc-text{font-size:1.1rem;font-weight:700;text-align:center;}"
        ".mc-dots span{opacity:.2;animation:mc-dot 1.2s infinite;}"
        ".mc-dots span:nth-child(2){animation-delay:.2s;}"
        ".mc-dots span:nth-child(3){animation-delay:.4s;}"
        "@keyframes mc-dot{0%,80%,100%{opacity:.2}40%{opacity:1}}"
        ".mc-bar{position:relative;width:230px;height:5px;border-radius:99px;background:#e0ecff;overflow:hidden;margin-top:4px;}"
        ".mc-bar::after{content:'';position:absolute;top:0;left:-40%;height:100%;width:40%;border-radius:99px;"
        "background:linear-gradient(90deg,#38bdf8,#2563eb);animation:mc-bar 1.4s ease-in-out infinite;}"
        "@keyframes mc-bar{0%{left:-40%}100%{left:100%}}"
        ".mc-term{font-family:Consolas,'Courier New',monospace;font-size:.78rem;color:#64748b;}"
        ".mc-caret{animation:mc-blink 1s steps(1) infinite;}"
        "@keyframes mc-blink{50%{opacity:0}}"
        # --- HUD: vòng radar xoay + tia quét phía sau thẻ ---
        ".mc-hud{position:absolute;top:50%;left:50%;width:0;height:0;z-index:1;pointer-events:none;}"
        ".mc-hud i{position:absolute;top:0;left:0;border-radius:50%;transform:translate(-50%,-50%);}"
        ".mc-hud .h1{width:520px;height:520px;border:2px dashed rgba(125,211,252,.35);animation:mc-spin 26s linear infinite;}"
        ".mc-hud .h2{width:680px;height:680px;border:1px solid rgba(125,211,252,.20);"
        "border-top-color:rgba(165,243,252,.85);animation:mc-spin 9s linear infinite reverse;}"
        ".mc-hud .h2::before{content:'';position:absolute;top:-6px;left:50%;margin-left:-5px;width:10px;height:10px;"
        "border-radius:50%;background:#7dd3fc;box-shadow:0 0 16px 4px rgba(56,189,248,.9);}"
        ".mc-hud .h3{width:880px;height:880px;border:2px dotted rgba(250,204,21,.28);animation:mc-spin 44s linear infinite;}"
        ".mc-hud .sw{width:780px;height:780px;"
        "background:conic-gradient(from 0deg,rgba(56,189,248,0) 0deg,rgba(56,189,248,.30) 42deg,rgba(56,189,248,0) 43deg);"
        "-webkit-mask-image:radial-gradient(circle,transparent 28%,#000 29%,#000 100%);"
        "mask-image:radial-gradient(circle,transparent 28%,#000 29%,#000 100%);animation:mc-spin 4.5s linear infinite;}"
        "@keyframes mc-spin{from{transform:translate(-50%,-50%) rotate(0deg)}to{transform:translate(-50%,-50%) rotate(360deg)}}"
        # --- bóng đổ dưới nhân vật co giãn theo nhịp nhún ---
        ".mc-shadow{width:96px;height:10px;border-radius:50%;background:rgba(15,23,42,.25);filter:blur(3px);"
        "margin-top:-6px;animation:mc-shadow 1.1s ease-in-out infinite;}"
        "@keyframes mc-shadow{0%,100%{transform:scaleX(1);opacity:.55}50%{transform:scaleX(.68);opacity:.28}}"
        # --- dòng lệnh luân phiên ---
        ".mc-ticker{position:relative;display:inline-block;width:300px;height:1.3em;vertical-align:bottom;}"
        ".mc-ticker span{position:absolute;left:0;width:100%;text-align:center;opacity:0;animation:mc-tick 10s infinite both;}"
        ".mc-ticker span::after{content:' ▌';animation:mc-blink 1s steps(1) infinite;}"
        ".mc-ticker span:nth-child(2){animation-delay:2.5s;}.mc-ticker span:nth-child(3){animation-delay:5s;}"
        ".mc-ticker span:nth-child(4){animation-delay:7.5s;}"
        "@keyframes mc-tick{0%{opacity:0;transform:translateY(7px)}3%,22%{opacity:1;transform:none}"
        "25%,100%{opacity:0;transform:translateY(-7px)}}"
        # --- pháo giấy khi xong ---
        ".mc-confetti{position:absolute;top:50%;left:50%;width:0;height:0;z-index:4;pointer-events:none;}"
        ".mc-confetti i{position:absolute;top:0;left:0;width:9px;height:15px;border-radius:2px;opacity:0;}"
        ".mc-closing .mc-confetti i{animation:mc-burst 1.15s cubic-bezier(.1,.8,.3,1) forwards;}"
        "@keyframes mc-burst{0%{opacity:1;transform:translate(0,0) rotate(0deg) scale(.4)}"
        "70%{opacity:1}100%{opacity:0;transform:translate(var(--dx),var(--dy)) rotate(var(--rot)) scale(1)}}"
        + _confetti_css() +
        # --- sonar: sóng lan tỏa từ tâm ---
        ".mc-hud .p{width:700px;height:700px;border:2px solid rgba(125,211,252,.55);opacity:0;animation:mc-ping 3.6s ease-out infinite;}"
        ".mc-hud .p2{animation-delay:1.2s;}.mc-hud .p3{animation-delay:2.4s;}"
        "@keyframes mc-ping{0%{transform:translate(-50%,-50%) scale(.28);opacity:.7}"
        "100%{transform:translate(-50%,-50%) scale(1.5);opacity:0}}"
        # --- dòng dữ liệu rơi ---
        ".mc-streams{position:absolute;top:0;left:0;width:100%;height:100%;overflow:hidden;pointer-events:none;}"
        ".mc-streams b{position:absolute;top:-30vh;width:2px;height:22vh;border-radius:2px;"
        "background:linear-gradient(transparent,rgba(125,211,252,.85));animation:mc-fall linear infinite;}"
        "@keyframes mc-fall{0%{transform:translateY(0);opacity:0}10%{opacity:.8}100%{transform:translateY(160vh);opacity:0}}"
        + _streams_css() +
        # --- thẻ nghiêng nhẹ kiểu 3D ---
        ".mc-tilt{position:relative;z-index:3;animation:mc-tilt 5s ease-in-out infinite alternate;}"
        "@keyframes mc-tilt{0%{transform:perspective(1000px) rotateX(3deg) rotateY(-6deg)}"
        "100%{transform:perspective(1000px) rotateX(-3deg) rotateY(6deg)}}"
        # --- thanh equalizer cạnh dòng chữ ---
        ".mc-eq{display:inline-flex;gap:3px;align-items:flex-end;height:16px;margin-right:9px;vertical-align:-2px;}"
        ".mc-eq i{width:3px;height:4px;border-radius:2px;background:linear-gradient(#38bdf8,#2563eb);"
        "animation:mc-eq .9s ease-in-out infinite;}"
        ".mc-eq i:nth-child(2){animation-delay:.15s}.mc-eq i:nth-child(3){animation-delay:.3s}"
        ".mc-eq i:nth-child(4){animation-delay:.45s}.mc-eq i:nth-child(5){animation-delay:.6s}"
        "@keyframes mc-eq{0%,100%{height:4px}50%{height:16px}}"
        # --- chữ thương hiệu nhấp nháy như tín hiệu ---
        ".mc-brand{animation:mc-flicker 5s infinite;}"
        "@keyframes mc-flicker{0%,91%,100%{opacity:1}92%{opacity:.25}93%{opacity:1}95%{opacity:.4}96%{opacity:1}}"
        # --- chuyển cảnh: trạng thái 'xong' rồi mờ dần (.mc-closing được JS thêm vào khi có kết quả) ---
        ".mc-state{display:flex;flex-direction:column;align-items:center;gap:8px;}"
        ".mc-finish,.mc-t2{display:none;}"
        ".mc-check{position:relative;width:44px;height:44px;border-radius:50%;"
        "background:radial-gradient(circle at 32% 28%,#bbf7d0,#22d3ee 52%,#2563eb 100%);"
        "box-shadow:0 0 18px rgba(34,211,238,.85),inset 0 0 8px rgba(255,255,255,.55);"
        "animation:mc-pop .55s cubic-bezier(.2,1.7,.4,1) both;}"
        ".mc-check::before{content:'';position:absolute;top:0;left:0;width:100%;height:100%;border-radius:50%;"
        "border:2px solid rgba(34,211,238,.9);animation:mc-ripple 1.3s ease-out .35s infinite;}"
        ".mc-check::after{content:'';position:absolute;left:16px;top:7px;width:11px;height:21px;"
        "border:solid #fff;border-width:0 4px 4px 0;border-radius:2px;transform:rotate(45deg) scale(0);"
        "filter:drop-shadow(0 0 4px rgba(255,255,255,.9));animation:mc-tick-draw .4s cubic-bezier(.3,1.6,.5,1) .3s forwards;}"
        "@keyframes mc-tick-draw{0%{transform:rotate(45deg) scale(0)}100%{transform:rotate(45deg) scale(1)}}"
        "@keyframes mc-ripple{0%{transform:scale(1);opacity:.9}100%{transform:scale(1.9);opacity:0}}"
        "@keyframes mc-pop{0%{transform:scale(0) rotate(-90deg)}100%{transform:scale(1) rotate(0)}}"
        ".mc-closing .mc-working,.mc-closing .mc-t1{display:none;}"
        ".mc-closing .mc-finish{display:flex;}"
        ".mc-closing .mc-t2{display:inline;}"
        ".mc-closing .mc-bar::after{animation:mc-bar-fill .55s ease-out forwards;}"
        "@keyframes mc-bar-fill{0%{left:-40%;width:40%}100%{left:0;width:100%}}"
        # Cổng không gian: nền bị "xé" thành lỗ tròn mở rộng từ tâm, viền neon + chớp sáng + nhiễu glitch trước đó
        "@property --r{syntax:'<percentage>';inherits:false;initial-value:-2%;}"
        ".mc-overlay.mc-closing{pointer-events:none;"
        "-webkit-mask-image:radial-gradient(circle at 50% 50%,transparent var(--r),#000 calc(var(--r) + 1%));"
        "mask-image:radial-gradient(circle at 50% 50%,transparent var(--r),#000 calc(var(--r) + 1%));"
        "animation:mc-portal 1.05s cubic-bezier(.65,0,.35,1) .95s both;}"
        "@keyframes mc-portal{0%{--r:-2%;opacity:1}88%{opacity:1}100%{--r:106%;opacity:0;visibility:hidden}}"
        ".mc-fx{position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:1000000;pointer-events:none;opacity:0;}"
        ".mc-overlay.mc-closing~.mc-fxflash{background:radial-gradient(circle at 50% 50%,rgba(255,255,255,.95),"
        "rgba(125,211,252,.55) 22%,transparent 55%);animation:mc-flash .6s ease-out .95s forwards;}"
        "@keyframes mc-flash{0%{opacity:0;transform:scale(.35)}22%{opacity:1}100%{opacity:0;transform:scale(1.7)}}"
        ".mc-overlay.mc-closing~.mc-fxring{background:radial-gradient(circle at 50% 50%,"
        "transparent calc(var(--r) - 1.4%),rgba(236,250,255,.98) calc(var(--r) - .2%),"
        "rgba(56,189,248,.7) calc(var(--r) + .9%),transparent calc(var(--r) + 3.6%));"
        "animation:mc-ringout 1.05s cubic-bezier(.65,0,.35,1) .95s forwards;}"
        "@keyframes mc-ringout{0%{--r:0%;opacity:1}85%{opacity:1}100%{--r:106%;opacity:0}}"
        ".mc-closing .mc-card{animation:mc-glitch .4s steps(1) .5s 1 both;}"
        "@keyframes mc-glitch{0%{transform:none;filter:none}"
        "15%{transform:translate(-5px,2px);filter:hue-rotate(40deg) saturate(2)}"
        "30%{transform:translate(6px,-2px) skewX(4deg);filter:hue-rotate(-60deg)}"
        "45%{transform:translate(-3px,0);filter:none}"
        "60%{transform:translate(4px,1px) skewX(-5deg);filter:hue-rotate(90deg) saturate(2.5)}"
        "80%{transform:translate(0,-2px);filter:none}100%{transform:none;filter:none}}"
        "@media (prefers-reduced-motion:reduce){.mc-overlay *,.mc-overlay{animation:none!important}}"
    )
    # Lưu ý: HTML viết liền, không xuống dòng/thụt lề để Markdown không biến thành code block
    return (
        f"<style>{css}</style>"
        '<div class="mc-overlay">'
        '<div class="mc-grid"></div><div class="mc-scan"></div><div class="mc-vignette"></div>'
        f'<div class="mc-streams">{"<b></b>" * 18}</div>'
        '<div class="mc-hud"><i class="p p1"></i><i class="p p2"></i><i class="p p3"></i><i class="h3"></i><i class="h1"></i><i class="h2"></i><i class="sw"></i></div>'
        f'<div class="mc-rain">{code_spans}</div>'
        f'<div class="mc-confetti">{"<i></i>" * _CONFETTI_N}</div>'
        '<div class="mc-brand">&lt;FITHCMUS /&gt; CNTT</div>'
        '<div class="mc-foot">NHẬP MÔN CÔNG NGHỆ THÔNG TIN · ĐH KHOA HỌC TỰ NHIÊN TP.HCM</div>'
        f'<div class="mc-tilt"><div class="mc-ring"><div class="mc-card">{img}<div class="mc-shadow"></div>'
        '<div class="mc-state mc-working">'
        '<div class="mc-runner"><span class="mc-a">🏃</span><span class="mc-b">🧑‍🦽</span></div>'
        f'<div class="mc-text"><span class="mc-eq"><i></i><i></i><i></i><i></i><i></i></span>{text}<span class="mc-dots"><span>.</span><span>.</span><span>.</span></span></div>'
        "</div>"
        '<div class="mc-state mc-finish"><div class="mc-check"></div>'
        '<div class="mc-text">Xong rồi! Kết quả của bạn đây</div></div>'
        '<div class="mc-bar"></div>'
        '<div class="mc-term"><span class="mc-t1"><span class="mc-ticker"><span>$ crawl --site</span>'
        '<span>$ compare --price --specs</span><span>$ rank --budget --purpose</span>'
        '<span>$ mcee --write-advice</span></span></span>'
        '<span class="mc-t2">$ mcee --done · exit 0<span class="mc-caret"> ▌</span></span></div>'
        "</div></div></div></div>"
        '<div class="mc-fx mc-fxflash"></div><div class="mc-fx mc-fxring"></div>'
    )


def show_loader(placeholder, text: str = f"{AI_NAME} đang phân tích và tìm kiếm") -> None:
    placeholder.markdown(loader_html(text), unsafe_allow_html=True)


# Hiệu ứng "trồi lên" lần lượt cho từng khối kết quả (chỉ áp dụng cho các khối con trực tiếp của khung kết quả)
_RESULT_SEL = (
    ".st-key-mc-results>[data-testid='stElementContainer'],"
    ".st-key-mc-results>[data-testid='element-container'],"
    ".st-key-mc-results>[data-testid='stVerticalBlock']>[data-testid='stElementContainer'],"
    ".st-key-mc-results>[data-testid='stVerticalBlock']>[data-testid='element-container']"
)
REVEAL_CSS = (
    "@keyframes mc-reveal{0%{opacity:0;transform:translateY(28px) scale(.985);filter:blur(6px)}"
    "100%{opacity:1;transform:none;filter:blur(0)}}"
    + _RESULT_SEL + "{animation:mc-reveal .85s cubic-bezier(.2,.7,.2,1) both;animation-delay:1.05s;}"
    + "".join(
        ",".join(f"{sel}:nth-child({i})" for sel in _RESULT_SEL.split(","))
        + f"{{animation-delay:{1.05 + 0.14 * (i - 1):.2f}s;}}"
        for i in range(2, 9)
    )
    + "@media (prefers-reduced-motion:reduce){.st-key-mc-results *{animation:none!important}}"
)


_TR = ".st-key-mc-results tbody tr"
_MD = ".st-key-mc-results [data-testid='stMarkdownContainer']"
APP_FX_CSS = (
    # --- tiêu đề: vạch neon co giãn ---
    "@keyframes mc-line{0%{width:60px}100%{width:240px}}"
    ".stApp h1::after{content:'';display:block;height:4px;width:120px;border-radius:99px;margin-top:10px;"
    "background:linear-gradient(90deg,#38bdf8,#2563eb,#facc15);animation:mc-line 2.6s ease-in-out infinite alternate;}"
    # --- nút chính: nhấc lên + tỏa sáng + vệt sáng quét ---
    ".stApp button[kind='primary'],.stApp [data-testid='stBaseButton-primary']"
    "{position:relative;overflow:hidden;transition:transform .2s ease,box-shadow .25s ease;}"
    ".stApp button[kind='primary']:hover,.stApp [data-testid='stBaseButton-primary']:hover"
    "{transform:translateY(-2px);box-shadow:0 10px 30px rgba(56,189,248,.45);}"
    ".stApp button[kind='primary']:active,.stApp [data-testid='stBaseButton-primary']:active"
    "{transform:translateY(0) scale(.99);}"
    ".stApp button[kind='primary']::after,.stApp [data-testid='stBaseButton-primary']::after"
    "{content:'';position:absolute;top:0;left:-60%;width:40%;height:100%;"
    "background:linear-gradient(100deg,transparent,rgba(255,255,255,.35),transparent);"
    "transform:skewX(-20deg);animation:mc-shine 3.2s ease-in-out infinite;}"
    "@keyframes mc-shine{0%,60%{left:-60%}100%{left:130%}}"
    # --- ô nhập: viền sáng khi focus ---
    ".stApp textarea:focus,.stApp input:focus"
    "{box-shadow:0 0 0 2px rgba(56,189,248,.55),0 0 18px rgba(56,189,248,.25)!important;}"
    # --- trang hiện dần khi mở ---
    "@keyframes mc-page{0%{opacity:0}100%{opacity:1}}"
    "header[data-testid='stHeader']{transition:opacity .5s ease;}"
    ".stApp [data-testid='stMainBlockContainer'],.stApp .block-container{animation:mc-page .8s ease-out;}"
    # --- bảng so sánh: hàng trượt vào lần lượt, rê chuột nổi bật ---
    ".st-key-mc-results table{border-collapse:separate;border-spacing:0;border-radius:12px;overflow:hidden;"
    "box-shadow:0 6px 24px rgba(15,23,42,.10);}"
    ".st-key-mc-results thead th{background:linear-gradient(90deg,rgba(56,189,248,.20),rgba(37,99,235,.12));}"
    + _TR + "{transition:background .2s ease,transform .2s ease;animation:mc-row .6s cubic-bezier(.2,.7,.2,1) both;"
    "animation-delay:1.5s;}"
    + "".join(f"{_TR}:nth-child({i}){{animation-delay:{1.5 + 0.09 * (i - 1):.2f}s;}}" for i in range(2, 13))
    + _TR + ":hover{background:rgba(56,189,248,.16);transform:translateX(5px);}"
    "@keyframes mc-row{0%{opacity:0;transform:translateX(-26px)}100%{opacity:1;transform:none}}"
    # --- liên kết: phát sáng khi rê chuột ---
    ".st-key-mc-results a{transition:text-shadow .2s ease;}"
    ".st-key-mc-results a:hover{text-shadow:0 0 10px rgba(56,189,248,.9);}"
    # --- tiêu đề mục: vạch neon mọc ra ---
    ".st-key-mc-results h3::after{content:'';display:block;height:3px;width:0;border-radius:99px;margin-top:4px;"
    "background:linear-gradient(90deg,#38bdf8,#2563eb,transparent);animation:mc-underline .9s ease-out 1.7s forwards;}"
    "@keyframes mc-underline{to{width:200px}}"
    # --- lời tư vấn của MCee hiện dần từng đoạn / từng ý ---
    + _MD + ">p," + _MD + " li{animation:mc-soft .7s ease-out both;animation-delay:1.8s;}"
    + "".join(f"{_MD} li:nth-child({i}){{animation-delay:{1.8 + 0.07 * (i - 1):.2f}s;}}" for i in range(2, 15))
    + "@keyframes mc-soft{0%{opacity:0;transform:translateY(10px)}100%{opacity:1;transform:none}}"
    "@media (prefers-reduced-motion:reduce){.st-key-mc-results *,.stApp h1::after,"
    ".stApp button[kind='primary']::after,.stApp [data-testid='stBaseButton-primary']::after{animation:none!important}}"
)


def results_container():
    """Khung chứa kết quả (có key để áp CSS hiệu ứng; phiên bản Streamlit cũ không hỗ trợ key thì dùng khung thường)."""
    try:
        return st.container(key="mc-results")
    except TypeError:
        return st.container()


def finish_overlay() -> None:
    """Báo cho lớp phủ chuyển sang trạng thái 'Xong rồi' + mờ dần, đồng thời cuộn mượt tới kết quả bên dưới."""
    try:
        import streamlit.components.v1 as components

        components.html(
            "<script>(function(){const d=window.parent.document;"
            "const ov=d.querySelector('.mc-overlay');if(ov){ov.classList.add('mc-closing');}"
            "const el=d.getElementById('ket-qua');"
            "if(el){setTimeout(function(){el.scrollIntoView({behavior:'smooth',block:'start'});},250);}"
            "})();</script>",
            height=0,
        )
    except Exception:
        pass


def _cell(x) -> str:
    return (str(x) if x else "—").replace("|", "/").replace("\n", " ")


def comparison_table_md(products: list[dict]) -> str:
    """Bảng so sánh dựng trực tiếp từ dữ liệu đã đọc (không phụ thuộc AI). Cột nào trống ở mọi máy thì ẩn đi."""
    cols = [("CPU", "cpu"), ("GPU", "gpu"), ("RAM", "ram"), ("RAM tối đa", "ram_max"), ("Ổ cứng", "ssd"),
            ("Màn hình", "screen"), ("Cân nặng", "weight"), ("HĐH", "os")]
    cols = [(label, key) for label, key in cols if any((p.get("specs") or {}).get(key) for p in products)]
    head = ["Tên máy", "Giá"] + [label for label, _ in cols] + ["Link"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for idx, s in enumerate(products):
        sp = s.get("specs") or {}
        price = f"{s['price']:,}".replace(",", ".") + " ₫"
        url = s["url"].replace("(", "%28").replace(")", "%29")
        name = ("🏆 " if idx == 0 and len(products) > 1 else "") + _cell(s["title"])
        cells = [name, f"**{price}**"] + [_cell(sp.get(key)) for _, key in cols] + [f"[Xem sản phẩm]({url})"]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


# =====================================================
# 5. GIAO DIỆN
# =====================================================
st.title("💻 Laptop Decision")
st.caption(f"{AI_NAME} – trợ lý AI tư vấn chọn laptop · Đồ án môn Nhập môn CNTT")
st.markdown(f"<style>{REVEAL_CSS}{APP_FX_CSS}</style>", unsafe_allow_html=True)

# Cài đặt nâng cao: sidebar được thu gọn sẵn, người dùng thường không cần mở.
with st.sidebar:
    st.header("⚙️ Cài đặt nâng cao")
    provider_name = st.selectbox("Nhà cung cấp AI", list(PROVIDERS.keys()), index=0)
    provider = PROVIDERS[provider_name]
    # key= giúp ô Model tự đổi giá trị mặc định khi đổi nhà cung cấp
    model_name = st.text_input("Model", value=provider["model"], key=f"model_{provider_name}")
    max_pages = st.slider("Số trang tối đa mỗi website", 3, 20, 20)
    total_chars = st.slider(
        "Tổng số ký tự gửi cho model", 6000, 60000, 60000, step=2000,
        help="Nếu AI trả lời quá chậm hoặc báo lỗi 413/429, hãy giảm số này.",
    )
    debug = st.checkbox("Chế độ kiểm tra (hiện nhật ký và nội dung trang)")
    st.caption("🌐 Website tìm kiếm mặc định (từ link.txt): " + ", ".join(host_of(u) for u in load_default_sites()))

API_URL = provider["url"]
API_KEY = load_secret(provider["key_name"]) if provider["key_name"] else ""
needs_key = bool(provider["key_name"])

if needs_key and not API_KEY:
    st.error(
        f"Chưa có {provider['key_name']}. Thêm vào file `.streamlit/secrets.toml` dòng "
        f'`{provider["key_name"]} = "..."` (hoặc mục Secrets khi deploy) rồi chạy lại.'
    )

st.subheader("1️⃣ Tiêu chí của bạn")
budget = st.slider("Ngân sách (triệu VNĐ)", 5, 500, (15, 25))
purpose = st.selectbox(
    "Mục đích sử dụng",
    ["Học tập – văn phòng", "Lập trình", "Đồ họa – thiết kế", "Chơi game", "Mỏng nhẹ – di động"],
)
c1, c2 = st.columns(2)
with c1:
    os_pref = st.selectbox("Hệ điều hành", ["Không quan tâm", "Windows", "macOS", "Linux/FreeDOS"])
with c2:
    brand_pref = st.text_input("Hãng laptop ưu tiên", placeholder="Asus, Dell, Lenovo...")
c3, c4, c5, c6 = st.columns(4)
with c3:
    cpu_brand = st.selectbox("Hãng CPU", CPU_BRAND_OPTIONS)
with c4:
    ram_pref = st.selectbox("RAM", RAM_OPTIONS)
with c5:
    gpu_pref = st.selectbox("GPU (card đồ họa)", GPU_OPTIONS)
with c6:
    ssd_pref = st.selectbox("Ổ cứng SSD", SSD_OPTIONS)
extra = st.text_area("Yêu cầu thêm", placeholder="VD: pin trâu, màn hình đẹp, dưới 2kg...", height=80)

submit_button = st.button("Tìm & tư vấn 🚀", type="primary", use_container_width=True)

# =====================================================
# 6. XỬ LÝ
# =====================================================
if submit_button:
    sites = load_default_sites()  # website lấy mặc định từ link.txt

    if needs_key and not API_KEY:
        st.stop()
    if not sites:
        st.warning("⚠️ Không đọc được danh sách website. Hãy kiểm tra file link.txt (mỗi dòng 1 link).")
        st.stop()

    criteria = (
        f"- Ngân sách: {budget[0]} – {budget[1]} triệu VNĐ\n"
        f"- Mục đích: {purpose}\n"
        f"- Hệ điều hành: {os_pref}\n"
        f"- Hãng ưu tiên: {brand_pref or 'Không có'}\n"
        f"- Hãng CPU: {cpu_brand}\n"
        f"- RAM: {ram_pref}\n"
        f"- GPU: {gpu_pref}\n"
        f"- Ổ cứng SSD: {ssd_pref}\n"
        f"- Yêu cầu thêm: {extra or 'Không có'}"
    )
    keywords = tuple(build_keywords(purpose, os_pref, brand_pref, cpu_brand, ram_pref, gpu_pref))
    lo, hi = budget[0] * 1_000_000, budget[1] * 1_000_000

    st.divider()
    st.markdown('<div id="ket-qua"></div>', unsafe_allow_html=True)
    results = results_container()  # khung kết quả nằm TRÊN lớp phủ trong luồng trang -> gỡ lớp phủ không làm trang giật
    loader = st.empty()
    show_loader(loader)

    sources: list[dict] = []
    all_logs: list[str] = []
    for site in sites:
        pages, logs = crawl_site(site, keywords, max_pages, lo, hi)
        sources.extend(pages)
        all_logs += [f"🌐 {site}"] + logs

    usable = [s for s in sources if s["ok"]]
    if not usable:
        loader.empty()
        with results:
            st.warning("Không đọc được nội dung từ website này nên chưa thể tư vấn. Hãy thử website khác.")
            if debug:
                st.code("\n".join(all_logs))
        st.stop()

    brands = [norm(b) for b in re.split(r"[,;/]+", brand_pref or "") if norm(b)]

    # Bỏ khỏi dữ liệu gửi cho AI: sản phẩm không phải laptop, hoặc có giá rõ ràng ngoài ngân sách (dung sai 10%)
    def irrelevant(s):
        if not s.get("is_product"):
            return False
        if not s.get("is_laptop"):
            return True
        if not spec_check(s.get("specs") or {}, cpu_brand, ram_pref, gpu_pref, ssd_pref)[0]:
            return True  # chắc chắn sai bộ lọc CPU/RAM/SSD/GPU
        return bool(s.get("price") and not (lo * 0.9 <= s["price"] <= hi * 1.1))

    relevant = [s for s in usable if not irrelevant(s)]
    if relevant:
        usable = relevant

    # Chia đều ngân sách ký tự cho các trang để không vượt giới hạn của model
    share = max(1500, total_chars // len(usable))
    usable = [{**s, "text": s["text"][:share]} for s in usable]

    # Phương án trong ngân sách: dựng bảng so sánh trực tiếp từ dữ liệu đã đọc
    products = [
        s for s in usable
        if s.get("is_product") and s.get("is_laptop") and s.get("price") and lo <= s["price"] <= hi
    ]
    products = [s for s in products if spec_check(s.get("specs") or {}, cpu_brand, ram_pref, gpu_pref, ssd_pref)[0]]
    products.sort(key=lambda s: (
        spec_check(s.get("specs") or {}, cpu_brand, ram_pref, gpu_pref, ssd_pref)[1],  # máy đã xác nhận đúng bộ lọc lên trước
        not any(b in norm(s["title"]) for b in brands),
        s["price"],
    ))
    products = products[:10]
    filters_on = any(x != "Không quan tâm" for x in (cpu_brand, ram_pref, gpu_pref, ssd_pref))
    unsure_n = sum(1 for s in products if spec_check(s.get("specs") or {}, cpu_brand, ram_pref, gpu_pref, ssd_pref)[1])

    # AI phân tích. Nếu AI lỗi thì im lặng, bảng so sánh vẫn là kết quả.
    chain = [{"url": API_URL, "key": API_KEY, "model": model_name}]
    for pname, prov in PROVIDERS.items():  # nhà cung cấp dự phòng: chỉ những nơi đã có key trong secrets
        if pname != provider_name and prov["key_name"] and "Hugging" not in pname and load_secret(prov["key_name"]):
            chain.append({"url": prov["url"], "key": load_secret(prov["key_name"]), "model": prov["model"]})
    answer, ai_error = ask_with_fallback(chain, criteria, usable)

    # ---------- Dựng kết quả NGAY DƯỚI lớp phủ (đang còn che kín) ----------
    with results:
        if products:
            st.subheader(f"🛒 So sánh {len(products)} phương án trong ngân sách")
            st.markdown(comparison_table_md(products))
            if filters_on and unsure_n:
                st.caption(
                    f"ℹ️ {unsure_n} máy chưa ghi rõ đủ thông tin CPU/RAM/SSD/GPU trên trang nên chưa xác nhận được "
                    "có đúng bộ lọc không (đã xếp sau các máy khớp rõ ràng). Bạn nên mở link để kiểm tra lại."
                )
        elif any(s["has_price"] for s in usable):
            st.info(
                f"{AI_NAME} chưa tìm thấy laptop nào có giá trong khoảng {budget[0]}–{budget[1]} triệu. "
                "Bạn thử nới ngân sách hoặc bỏ bớt bộ lọc (hãng, CPU, RAM, SSD, GPU) nhé."
            )
        else:
            st.warning(
                "Không tìm thấy giá sản phẩm trên website này (có thể website chặn truy cập tự động "
                "hoặc tải sản phẩm bằng JavaScript). Hãy thử website khác."
            )

        if answer:
            st.subheader(f"💬 {AI_NAME} tư vấn")
            st.markdown(answer)
        else:
            # AI chưa phản hồi được: luôn có nhận xét tự động từ bảng để phần này không bị trống
            if products:
                st.subheader(f"💬 {AI_NAME} nhận xét nhanh")
                st.markdown(quick_review(products, purpose, brand_pref, extra))
                st.caption(
                    "Phần phân tích chi tiết của MCee đang bận, bạn thử lại sau ít phút nhé. "
                    "Nhận xét trên được tổng hợp tự động từ bảng so sánh."
                )
            if debug and ai_error:
                st.caption(f"[kiểm tra] Lỗi gọi AI: {ai_error}")

        # Nguồn dữ liệu: gấp gọn sẵn
        with st.expander(f"📚 Nguồn dữ liệu ({len(usable)} trang)"):
            for i, s in enumerate(usable, start=1):
                st.markdown(f"{i}. [{s['title']}]({s['url']})")
            if debug:
                st.markdown("**Dòng liên quan cấu hình (để chỉnh bộ nhận diện)**")
                for s in products:
                    keep = [
                        ln for ln in s["text"].splitlines()
                        if re.search(r"cpu|vi xử lý|intel|amd|ryzen|core|windows|win\s?1|ram|ssd|inch", ln, re.I)
                    ][:8]
                    st.code(s["title"] + "\n" + "\n".join(keep) + "\n=> " + str(s.get("specs")))
                st.markdown("**Nhật ký khám phá**")
                st.code("\n".join(all_logs))
                for i, s in enumerate(usable, start=1):
                    st.markdown(f"**[Nguồn {i}] {s['title']}** — {len(s['text'])} ký tự")
                    st.text(s["text"])

    # ---------- Chuyển cảnh: 'Xong rồi!' -> lớp phủ mờ dần -> kết quả trồi lên lần lượt ----------
    finish_overlay()
    time.sleep(2.9)  # chờ hiệu ứng mờ dần chạy xong rồi mới gỡ lớp phủ (cũng là phương án an toàn nếu JS không chạy)
    loader.empty()