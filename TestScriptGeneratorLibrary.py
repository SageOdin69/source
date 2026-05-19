"""
file: TestScriptGeneratorLibrary.py

Pre-test automation utility that converts proprietary .tst scripts
(PDL + UP/UM statements) into executable pytest scripts that are
fully compatible with CyfastFapLibrary.

Generated code follows the same pattern used in hand-written tests:
    cyfastobj.find_element_if_present_visual(ref_img, request)
    cyfastobj.find_element_and_click_visual(ref_img, request)
    cyfastobj.get_screenshot_and_verify_text(text, roi, request)

Pipeline:
    1.  Parse source .tst  ->  structured TestScript
    2.  OCR all supplied reference images  ->  element coordinate map
    3.  For each UP/UM statement, auto-select the best reference image
        and crop the relevant UI element, saving it under the project path
    4.  Emit a ready-to-run pytest .py file

Dependencies (all open-source, self-hosted):
    pip install rapidocr-onnxruntime opencv-python pillow numpy

Optional VLM-assisted cropping (better for complex layouts):
    pip install ollama
    Pull any vision model:  ollama pull qwen2-vl
                            ollama pull llama3.2-vision
                            ollama pull minicpm-v
"""

from __future__ import annotations

import json
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Optional: Ollama VLM
# ---------------------------------------------------------------------------
try:
    import ollama as _ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional: RapidOCR
# ---------------------------------------------------------------------------
try:
    from rapidocr_onnxruntime import RapidOCR as _RapidOCR
    _RAPIDOCR_AVAILABLE = True
except ImportError:
    _RAPIDOCR_AVAILABLE = False
    warnings.warn(
        "rapidocr-onnxruntime not installed. "
        "Install with: pip install rapidocr-onnxruntime",
        stacklevel=2,
    )

DEFAULT_VLM_MODEL = "qwen2-vl"
DEFAULT_VLM_HOST  = "http://localhost:11434"
CROP_PADDING      = 15
MIN_CROP_SIZE     = 20


# =============================================================================
# Data model
# =============================================================================

class UPAction(Enum):
    VERIFY_VISIBLE         = auto()
    VERIFY_AND_CLICK       = auto()
    VERIFY_STATE           = auto()
    CLICK_VERIFY_INVISIBLE = auto()
    VERIFY_TEXT_PRESENT    = auto()
    PROCEED                = auto()
    UNKNOWN                = auto()


@dataclass
class OCRHit:
    text: str
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def padded(self, px: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        return (
            max(0,     self.x1 - px),
            max(0,     self.y1 - px),
            min(img_w, self.x2 + px),
            min(img_h, self.y2 + px),
        )


@dataclass
class RefImageInfo:
    path:     str
    label:    str
    ocr_hits: List[OCRHit]
    width:    int
    height:   int

    def find(self, keyword: str, min_conf: float = 0.5) -> Optional[OCRHit]:
        kw = keyword.lower()
        exact = [h for h in self.ocr_hits if kw in h.text.lower() and h.conf >= min_conf]
        if exact:
            return max(exact, key=lambda h: h.conf)
        # fuzzy
        ranked = sorted(
            self.ocr_hits,
            key=lambda h: SequenceMatcher(None, kw, h.text.lower()).ratio(),
            reverse=True,
        )
        if ranked and SequenceMatcher(None, kw, ranked[0].text.lower()).ratio() >= 0.6:
            return ranked[0]
        return None

    def text_score(self, needle: str) -> float:
        words = re.findall(r"\w+", needle.lower())
        if not words:
            return 0.0
        all_text = " ".join(h.text.lower() for h in self.ocr_hits)
        return sum(1 for w in words if w in all_text) / len(words)


@dataclass
class UPStatement:
    raw:             str
    action:          UPAction       = UPAction.UNKNOWN
    target_element:  Optional[str]  = None
    click_element:   Optional[str]  = None
    state_value:     Optional[str]  = None
    ref_image:       Optional[str]  = None
    crop_path:       Optional[str]  = None
    roi_coord:       Optional[Tuple] = None
    click_crop_path: Optional[str]  = None
    click_roi:       Optional[Tuple] = None


@dataclass
class UMStatement:
    raw:          str
    ref_image:    Optional[str] = None
    crop_paths:   List[str]     = field(default_factory=list)


@dataclass
class PDLLine:
    raw: str


@dataclass
class TestScript:
    name:          str
    pdl_lines:     List[PDLLine]     = field(default_factory=list)
    includes:      List[str]         = field(default_factory=list)
    we_statements: List[str]         = field(default_factory=list)
    um_statements: List[UMStatement] = field(default_factory=list)
    up_statements: List[UPStatement] = field(default_factory=list)


# =============================================================================
# Script Parser
# =============================================================================

class ScriptParser:
    _BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
    _INCLUDE       = re.compile(r'^include\s+"(.+?)"', re.I)
    _WE            = re.compile(r"^WE\s+(.+)$", re.I)
    _UP            = re.compile(r'^UP\s+"?(.*?)"?\s*$')
    _UM            = re.compile(r'^UM\s+"?(.*?)"?\s*$')
    _TITLE         = re.compile(r'^title\s+', re.I)

    def parse(self, path: str) -> TestScript:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
        raw = self._BLOCK_COMMENT.sub("", raw)
        script = TestScript(name=Path(path).stem)

        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            m = self._INCLUDE.match(line)
            if m:
                script.includes.append(m.group(1))
                continue

            m = self._WE.match(line)
            if m:
                script.we_statements.append(m.group(1).strip())
                continue

            if self._TITLE.match(line):
                continue

            m = self._UP.match(line)
            if m:
                script.up_statements.append(UPStatement(raw=m.group(1).strip()))
                continue

            m = self._UM.match(line)
            if m:
                script.um_statements.append(UMStatement(raw=m.group(1).strip()))
                continue

            tokens = line.split()
            if tokens and tokens[0] in ("P", "S", "E", "D", "T", "R", "EP"):
                script.pdl_lines.append(PDLLine(raw=line))

        return script


# =============================================================================
# UP Intent Classifier
# =============================================================================

class UPIntentClassifier:
    _RULES: List[Tuple[re.Pattern, UPAction]] = [
        # click X -> verify Y invisible
        (re.compile(
            r"click\s+(?:on\s+)?.+?[\.,]\s*verify.{0,60}"
            r"(?:invisible|removed|not\s+visible|disappear|gone)",
            re.I), UPAction.CLICK_VERIFY_INVISIBLE),

        # verify X invisible
        (re.compile(
            r"(?:verify|check|confirm)\s+.{0,80}"
            r"(?:invisible|removed|not\s+visible|hidden|disappear|gone)",
            re.I), UPAction.VERIFY_VISIBLE),

        # verify X indicated as Y / state
        (re.compile(
            r"(?:verify|check|confirm)\s+.{0,80}"
            r"(?:indicated?\s+as|shown?\s+as|displays?\s+|set\s+to|reads?\s+|marked\s+as)\s*"
            r"['\"]?([A-Za-z0-9_\-\.\s]+?)['\"]?(?:\s*[\.\n]|$)",
            re.I), UPAction.VERIFY_STATE),

        # verify visible + "proceed to continue" -> also click
        (re.compile(
            r"(?:verify|check)\s+.{0,100}"
            r"(?:visible|present|exists?|appears?).{0,40}proceed\s+to\s+continue",
            re.I), UPAction.VERIFY_AND_CLICK),

        # verify visible
        (re.compile(
            r"(?:verify|check|confirm|ensure)\s+.{0,80}"
            r"(?:visible|present|exists?|appears?|shown|displayed)",
            re.I), UPAction.VERIFY_VISIBLE),

        # proceed / finish
        (re.compile(
            r"^(?:proceed|script\s+\S+\s+finished?|finish)",
            re.I), UPAction.PROCEED),
    ]

    @classmethod
    def classify(cls, raw: str) -> Tuple[UPAction, Optional[str], Optional[str], Optional[str]]:
        text = raw.strip().strip('"\'')

        for pattern, action in cls._RULES:
            m = pattern.search(text)
            if m:
                state_val  = None
                click_elem = None

                if action == UPAction.VERIFY_STATE and m.lastindex:
                    state_val = m.group(m.lastindex).strip().rstrip(".")

                if action == UPAction.CLICK_VERIFY_INVISIBLE:
                    # Extract what to click: "click on <X>"
                    cm = re.search(r"click\s+(?:on\s+)?(.+?)[\.,]", text, re.I)
                    if cm:
                        click_elem = cls._clean(cm.group(1))
                    # For target (what to check invisible), extract after "verify"
                    vm = re.search(r"verify\s+(?:that\s+)?(.+?)"
                                   r"(?:invisible|removed|not\s+visible)", text, re.I)
                    target = cls._clean(vm.group(1).strip()) if vm else cls._extract_target(text)
                    return action, target, click_elem, state_val

                target = cls._extract_target(text)
                return action, target, click_elem, state_val

        if re.search(r"\b(?:verify|check|confirm|ensure)\b", text, re.I):
            return UPAction.VERIFY_TEXT_PRESENT, cls._extract_target(text), None, None

        return UPAction.UNKNOWN, None, None, None

    @classmethod
    def _extract_target(cls, text: str) -> Optional[str]:
        t = re.sub(
            r"^(?:verify|check|confirm|ensure|click|press|select|proceed|script)"
            r"\s+(?:that\s+)?(?:on\s+)?(?:a\s+|an\s+|the\s+)?",
            "", text.strip(), flags=re.I,
        )
        parts = re.split(r"\s+(?:is\s+|are\s+|has\s+|should\s+)", t, 1, re.I)
        cand  = parts[0].strip()
        cand  = re.sub(r"\s+(?:in\s+the|in|on\s+the|of\s+the).{0,40}$", "", cand, flags=re.I)
        cand  = re.sub(r"[.,:;!?\n\\]+$", "", cand).strip()
        return cls._clean(cand) or None

    @staticmethod
    def _clean(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip())[:60]


# =============================================================================
# OCR Engine
# =============================================================================

class ImageOCREngine:
    def __init__(self):
        self._cache: Dict[str, RefImageInfo] = {}
        self._ocr = _RapidOCR() if _RAPIDOCR_AVAILABLE else None

    def process(self, image_path: str, label: str = "") -> RefImageInfo:
        key = str(Path(image_path).resolve())
        if key in self._cache:
            return self._cache[key]

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        h, w = img.shape[:2]

        hits: List[OCRHit] = []
        if self._ocr:
            result, _ = self._ocr(img)
            if result:
                for item in result:
                    box, text, conf = item[0], item[1], float(item[2])
                    xs = [int(p[0]) for p in box]
                    ys = [int(p[1]) for p in box]
                    hits.append(OCRHit(
                        text=text,
                        x1=min(xs), y1=min(ys),
                        x2=max(xs), y2=max(ys),
                        conf=conf,
                    ))

        info = RefImageInfo(path=image_path,
                            label=label or Path(image_path).stem,
                            ocr_hits=hits, width=w, height=h)
        self._cache[key] = info
        return info


# =============================================================================
# Element Cropper
# =============================================================================

class ElementCropper:
    def __init__(self, ocr_engine, use_vlm=False,
                 vlm_model=DEFAULT_VLM_MODEL, vlm_host=DEFAULT_VLM_HOST,
                 crop_padding=CROP_PADDING):
        self._ocr     = ocr_engine
        self._use_vlm = use_vlm and _OLLAMA_AVAILABLE
        self._vlm_model = vlm_model
        self._vlm_host  = vlm_host
        self._pad       = crop_padding

    def crop_element(self, image_path: str, description: str,
                     save_path: str,
                     search_keywords: Optional[List[str]] = None
                     ) -> Optional[Tuple[int, int, int, int]]:
        info = self._ocr.process(image_path)
        roi  = None

        if self._use_vlm:
            roi = self._vlm_locate(image_path, description)

        if roi is None:
            kws = search_keywords or self._keywords(description)
            roi = self._ocr_locate(info, kws)

        if roi is None:
            print(f"  [WARN] No region for '{description[:50]}' — using full image")
            roi = (0, 0, info.width, info.height)

        self._save(image_path, roi, save_path)
        return roi

    def _ocr_locate(self, info: RefImageInfo,
                    keywords: List[str]) -> Optional[Tuple[int, int, int, int]]:
        hits = [h for kw in keywords for h in [info.find(kw)] if h]
        if not hits:
            return None
        best = max(hits, key=lambda h: h.conf)
        x1, y1, x2, y2 = best.x1, best.y1, best.x2, best.y2
        for h in hits:
            if h is not best:
                dist = abs(h.center[0]-best.center[0]) + abs(h.center[1]-best.center[1])
                if dist < 120:
                    x1 = min(x1, h.x1); y1 = min(y1, h.y1)
                    x2 = max(x2, h.x2); y2 = max(y2, h.y2)
        p = self._pad
        return (max(0, x1-p), max(0, y1-p),
                min(info.width, x2+p), min(info.height, y2+p))

    def _vlm_locate(self, image_path: str, description: str
                    ) -> Optional[Tuple[int, int, int, int]]:
        if not _OLLAMA_AVAILABLE:
            return None
        prompt = (
            "Locate the UI element described below in this aviation HSI screenshot. "
            'Return ONLY JSON: {"x1":<int>,"y1":<int>,"x2":<int>,"y2":<int>}\n\n'
            f'Element: "{description}"\nNo explanation.'
        )
        try:
            client = _ollama.Client(host=self._vlm_host)
            resp   = client.chat(model=self._vlm_model,
                                 messages=[{"role": "user", "content": prompt,
                                            "images": [image_path]}])
            raw  = re.sub(r"```(?:json)?", "", resp["message"]["content"]).strip().rstrip("`")
            data = json.loads(raw)
            return int(data["x1"]), int(data["y1"]), int(data["x2"]), int(data["y2"])
        except Exception as e:
            print(f"  [WARN] VLM failed: {e}")
            return None

    @staticmethod
    def _keywords(text: str) -> List[str]:
        # General stop words — NOTE: keep domain UI element names like "Select"
        stop = {
            "the","a","an","is","are","on","in","that","this","to","of","and",
            "or","for","be","at","as","it","its","with","from","by","was","were",
            "verify","check","ensure","confirm","click","proceed",
            "panel","button","visible","invisible","present","removed","section",
            "different","twelve","opens","having","opening","clicking",
            "please","open","fly","out","menu","page","control","entire",
        }
        words = re.findall(r"[A-Za-z][A-Za-z0-9]+", text)
        return [w for w in words if w.lower() not in stop and len(w) > 2]

    @staticmethod
    def _save(image_path: str, roi: Tuple, save_path: str) -> None:
        img = cv2.imread(image_path)
        if img is None:
            return
        x1, y1, x2, y2 = roi
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2] if (x2-x1) >= MIN_CROP_SIZE and (y2-y1) >= MIN_CROP_SIZE else img
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, crop)
        print(f"  [CROP] {Path(save_path).name}  roi={roi}")


# =============================================================================
# Ref Image Assigner
# =============================================================================

class RefImageAssigner:
    def __init__(self, ocr_engine: ImageOCREngine):
        self._ocr = ocr_engine

    def assign(self, statements: List, ref_images: List[RefImageInfo],
               explicit_map: Optional[Dict[int, str]] = None) -> None:
        if not ref_images:
            return
        for idx, stmt in enumerate(statements):
            if explicit_map and idx in explicit_map:
                stmt.ref_image = explicit_map[idx]
                continue
            scores  = [info.text_score(stmt.raw) for info in ref_images]
            best    = int(np.argmax(scores))
            stmt.ref_image = ref_images[best].path


# =============================================================================
# Pytest Code Generator
# =============================================================================

class PytestCodeGenerator:
    """
    Emits pytest code that matches the pattern of test_tc_fap_04_5005.py exactly.
    """

    def generate(self, script: TestScript, output_path: str) -> str:
        lines: List[str] = []
        lines += self._header(script.name)
        lines += self._body(script)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
        return output_path

    # -------------------------------------------------------------------------

    def _header(self, name: str) -> List[str]:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        return [
            f"# Auto-generated pytest script for: {name}",
            f"# Generated by TestScriptGeneratorLibrary  {ts}",
            "# DO NOT EDIT MANUALLY — re-generate from the source .txt script.",
            "",
            "import time",
            "",
        ]

    def _body(self, script: TestScript) -> List[str]:
        lines = [f"def test_{script.name}(cyfastobj, request):"]

        # includes
        for inc in script.includes:
            m = self._inc_method(inc)
            if m:
                lines.append(f"\tcyfastobj.{m}()")

        # PDL
        lines.append("\t# PDL: drive HSI to required state")
        lines += self._pdl(script.pdl_lines)

        # UM
        for um in script.um_statements:
            lines.append(f"\t#To be Done -> UM \"{um.raw}\"")
            for cp in um.crop_paths:
                lines.append(f"\tcyfastobj.find_element_and_click_visual(\"{cp}\", request)")

        # UP
        for idx, up in enumerate(script.up_statements):
            lines += self._up(up, idx)

        return lines

    def _pdl(self, pdl_lines: List[PDLLine]) -> List[str]:
        lines = []
        mmap  = {"S": "set_state", "E": "set_extended_state",
                 "D": "set_decimal_state", "T": "set_text_state", "R": "set_r_state"}
        for pl in pdl_lines:
            t = pl.raw.split()
            if t[0] == "P":
                pid = t[1] if len(t) > 1 else "0x01"
                lines.append(f"\tcyfastobj.set_page('{pid}')")
            elif t[0] in mmap and len(t) >= 3:
                lines.append(f"\tcyfastobj.{mmap[t[0]]}('{t[1]}','{t[2]}')")
            elif t[0] == "EP":
                lines.append("\tcyfastobj.end_page()")
        return lines

    def _up(self, up: UPStatement, idx: int) -> List[str]:
        lines = [f"\t#To be Done -> UP \"{up.raw}\""]

        if up.action in (UPAction.PROCEED, UPAction.UNKNOWN):
            return lines

        crop   = up.crop_path  or "None"
        roi    = list(up.roi_coord) if up.roi_coord else None
        target = up.target_element or up.raw[:40]
        fname  = Path(crop).name if crop != "None" else "element"

        # ── VERIFY_AND_CLICK ─────────────────────────────────────────────
        if up.action == UPAction.VERIFY_AND_CLICK:
            if roi:
                lines.append(f"\texpected_text_roi = {roi}")
                lines.append(
                    f"\tassert cyfastobj.get_screenshot_and_verify_text"
                    f"(\"{self._short(target)}\", expected_text_roi, request),"
                    f" \"{target[:40]} is not present\""
                )
            lines.append(
                f"\tret_val = cyfastobj.find_element_if_present_visual"
                f"(\"{crop}\", request) #ToBeChecked"
            )
            lines.append("\tif ret_val is not None:")
            lines.append("\t\tx, y = ret_val")
            if roi:
                lines.append(f"\t\tassert x > {roi[0]} and y > {roi[1]},"
                             f" \"{target[:40]} not enabled/present\"")
            lines.append("\telse:")
            lines.append(f"\t\tassert False, \"{fname} is not found\"")
            lines.append(f"\tcyfastobj.find_element_and_click_visual(\"{crop}\", request)")

        # ── VERIFY_VISIBLE ────────────────────────────────────────────────
        elif up.action == UPAction.VERIFY_VISIBLE:
            lines.append(
                f"\tret_val = cyfastobj.find_element_if_present_visual"
                f"(\"{crop}\", request) #ToBeChecked"
            )
            lines.append("\tif ret_val is not None:")
            lines.append("\t\tx, y = ret_val")
            if roi:
                lines.append(f"\t\tassert x > {roi[0]} and y > {roi[1]},"
                             f" \"{target[:40]} is not present\"")
            lines.append("\telse:")
            lines.append(f"\t\tassert False, \"{fname} is not found\"")

        # ── VERIFY_STATE ─────────────────────────────────────────────────
        elif up.action == UPAction.VERIFY_STATE:
            sv = up.state_value or target
            if roi:
                lines.append(
                    f"\t#assert cyfastobj.get_screenshot_and_verify_text"
                    f"(\"{sv}\",[{roi[0]},{roi[1]},{roi[2]},{roi[3]}],request),"
                    f" \"{target[:40]} Running text is not present\""
                )
            lines.append(
                f"\tret_val = cyfastobj.find_element_if_present_visual"
                f"(\"{crop}\", request) #ToBeChecked"
            )
            lines.append("\tif ret_val is not None:")
            lines.append("\t\tx, y = ret_val")
            if roi:
                lines.append(f"\t\t#assert x > {roi[0]} and y > {roi[1]},"
                             f" \"{target[:40]} not in expected state\"")
            lines.append("\telse:")
            lines.append(f"\t\tassert False, \"{fname} is not found\"")

        # ── CLICK_VERIFY_INVISIBLE ────────────────────────────────────────
        elif up.action == UPAction.CLICK_VERIFY_INVISIBLE:
            click_crop = up.click_crop_path or "None"
            lines.append(
                f"\tcyfastobj.find_element_and_click_visual(\"{click_crop}\", request)"
            )
            lines.append(
                f"\tret_val = cyfastobj.find_element_if_present_visual"
                f"(\"{crop}\", request) #ToBeChecked"
            )
            lines.append("\tif ret_val is None:")
            lines.append(f"\t\t#assert x > {roi[0] if roi else 0} and y > {roi[1] if roi else 0},"
                         f" \"{target[:40]} is not present\"")
            lines.append("\telse:")
            lines.append(f"\t\tassert False, \"{fname} is not found\"")

        # ── VERIFY_TEXT_PRESENT ───────────────────────────────────────────
        elif up.action == UPAction.VERIFY_TEXT_PRESENT:
            if roi:
                lines.append(f"\texpected_text_roi = {roi}")
                lines.append(
                    f"\tassert cyfastobj.get_screenshot_and_verify_text"
                    f"(\"{self._short(target)}\", expected_text_roi, request),"
                    f" \"{target[:40]} text not present\""
                )
            else:
                lines.append(
                    f"\tret_val = cyfastobj.find_element_if_present_visual(\"{crop}\", request)"
                )
                lines.append("\tif ret_val is None:")
                lines.append(f"\t\tassert False, \"{target[:40]} not found\"")

        return lines

    @staticmethod
    def _short(s: str) -> str:
        """First meaningful word — used as text search key."""
        words = re.findall(r"[A-Z][a-z]+|[A-Za-z]{3,}", s)
        return words[0] if words else s[:15]

    @staticmethod
    def _inc_method(inc: str) -> Optional[str]:
        return {
            "cam-layout-1.tst":  "cam_layout_1",
            "cam-layout-2.tst":  "cam_layout_2",
            "cam-layout-3.tst":  "cam_layout_3",
            "cam-layout-4.tst":  "cam_layout_4",
            "cam-layout-5.tst":  "cam_layout_5",
            "tc_fap_global_1.tst": "tc_fap_global_1",
        }.get(inc.lower())


# =============================================================================
# Main Facade
# =============================================================================

class TestScriptGeneratorLibrary:
    """
    Top-level API.

    Basic usage
    -----------
    gen = TestScriptGeneratorLibrary()
    gen.generate(
        script_path      = "tc_fap_04_5005.txt",
        reference_images = [
            ("1.png", "master_page"),
            ("2.png", "flyout_expanded"),
            ("3.png", "cabin_lighting_main"),
            ("4.png", "entire_scenario_dialog"),
        ],
        output_path    = "generated_tests/test_tc_fap_04_5005.py",
        ref_output_dir = "./ref/CIL_/",
    )

    With explicit UP -> image mapping
    ----------------------------------
    gen.generate(
        ...,
        up_image_map = {
            0: "3.png",   # UP[0] -> Image 3 (Select button)
            1: "4.png",   # UP[1] -> Image 4 (12 scenarios dialog)
            2: "4.png",   # UP[2] -> Image 4 (Breakfast Run)
            3: "4.png",   # UP[3] -> Image 4 (Cancel)
        },
    )

    With VLM (Ollama) for better crop accuracy
    -------------------------------------------
    gen = TestScriptGeneratorLibrary(use_vlm=True, vlm_model="qwen2-vl")
    """

    def __init__(self, use_vlm=False, vlm_model=DEFAULT_VLM_MODEL,
                 vlm_host=DEFAULT_VLM_HOST, crop_padding=CROP_PADDING):
        self._ocr_engine = ImageOCREngine()
        self._cropper    = ElementCropper(self._ocr_engine, use_vlm,
                                          vlm_model, vlm_host, crop_padding)
        self._assigner   = RefImageAssigner(self._ocr_engine)
        self._generator  = PytestCodeGenerator()
        self._parser     = ScriptParser()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def generate(
        self,
        script_path:      str,
        reference_images: List[Tuple[str, str]],
        output_path:      Optional[str]           = None,
        ref_output_dir:   str                     = "./ref/CIL_/",
        up_image_map:     Optional[Dict[int, str]] = None,
        um_image_map:     Optional[Dict[int, str]] = None,
    ) -> str:
        name = Path(script_path).stem
        print(f"\n{'='*62}\n  TestScriptGeneratorLibrary  —  {name}\n{'='*62}")

        # 1. Parse
        print("\n[1/4] Parsing script …")
        script = self._parser.parse(script_path)
        print(f"      includes={len(script.includes)}  "
              f"pdl={len(script.pdl_lines)}  "
              f"UM={len(script.um_statements)}  "
              f"UP={len(script.up_statements)}")

        # 2. Classify intents
        print("\n[2/4] Classifying UP intents …")
        for up in script.up_statements:
            act, tgt, click, sv = UPIntentClassifier.classify(up.raw)
            up.action, up.target_element = act, tgt
            up.click_element, up.state_value = click, sv
            print(f"  [{act.name:<25}] {up.raw[:60]}")

        # 3. OCR reference images
        print("\n[3/4] OCR & assigning reference images …")
        ref_infos: List[RefImageInfo] = []
        for img_path, label in reference_images:
            info = self._ocr_engine.process(img_path, label)
            ref_infos.append(info)
            print(f"  {label}: {len(info.ocr_hits)} regions  ({img_path})")

        self._assigner.assign(script.up_statements, ref_infos, up_image_map)
        self._assigner.assign(script.um_statements, ref_infos, um_image_map)

        # Print assignment summary
        print("\n  UP → image assignment:")
        for i, up in enumerate(script.up_statements):
            img_lbl = next((l for p, l in reference_images
                            if p == up.ref_image), "?")
            print(f"    UP[{i}] → {img_lbl}  |  target: {up.target_element}")

        # 4. Crop elements
        print("\n[4/4] Cropping elements …")
        ref_dir = Path(ref_output_dir)
        ref_dir.mkdir(parents=True, exist_ok=True)

        for idx, up in enumerate(script.up_statements):
            if up.action in (UPAction.PROCEED, UPAction.UNKNOWN) or not up.ref_image:
                continue

            safe = re.sub(r"[^\w]", "_", (up.target_element or f"up{idx}"))[:30]
            crop_path = str(ref_dir / f"{safe}.png")

            # Choose the tightest, most relevant keywords per action
            if up.action == UPAction.VERIFY_AND_CLICK:
                # e.g. "Select Button visible" → just look for "Select"
                kws = [w for w in ElementCropper._keywords(up.target_element or up.raw)
                       if len(w) > 3][:2]
            elif up.action == UPAction.VERIFY_STATE:
                # Use ONLY the expected state text (e.g. "Run") as search keyword.
                # The target element name (e.g. "Breakfast") may not be in the image.
                kws = [up.state_value] if up.state_value else \
                      [w for w in ElementCropper._keywords(up.target_element or up.raw)
                       if len(w) > 3][:2]
            elif up.action == UPAction.CLICK_VERIFY_INVISIBLE:
                # target = what should go invisible → look for panel/dialog keywords
                kws = [w for w in ElementCropper._keywords(up.target_element or up.raw)
                       if len(w) > 4][:2]
            else:
                kws = ElementCropper._keywords(up.target_element or up.raw)[:3]

            roi = self._cropper.crop_element(up.ref_image,
                                             up.target_element or up.raw,
                                             crop_path, kws)
            up.crop_path = crop_path
            up.roi_coord = roi

            # click element crop for CLICK_VERIFY_INVISIBLE
            if up.action == UPAction.CLICK_VERIFY_INVISIBLE and up.click_element:
                safe_c = re.sub(r"[^\w]", "_", up.click_element)[:30]
                c_path = str(ref_dir / f"{safe_c}.png")
                click_kws = [w for w in ElementCropper._keywords(up.click_element)
                             if len(w) > 2]
                self._cropper.crop_element(up.ref_image, up.click_element,
                                           c_path, click_kws)
                up.click_crop_path = c_path

        # UM crops — generate meaningful click targets only
        for um in script.um_statements:
            if not um.ref_image:
                continue
            # Use only the last 2 strong nouns from the UM as click targets
            kws = ElementCropper._keywords(um.raw)
            # Filter to words >4 chars that are likely element names
            strong = [w for w in kws if len(w) > 4][:2]
            for kw in strong:
                safe_k = re.sub(r"[^\w]", "_", kw)[:30]
                cp = str(ref_dir / f"{safe_k}.png")
                self._cropper.crop_element(um.ref_image, kw, cp, [kw])
                um.crop_paths.append(cp)

        # 5. Generate
        if output_path is None:
            output_path = f"generated_tests/test_{name}.py"
        out = self._generator.generate(script, output_path)
        print(f"\n{'='*62}")
        print(f"  Done.  pytest file  →  {out}")
        print(f"  Ref crops stored   →  {ref_output_dir}")
        print(f"{'='*62}\n")
        return out

    def parse_and_summarise(self, script_path: str) -> TestScript:
        """Parse + print a classification summary without generating any files."""
        script = self._parser.parse(script_path)
        for up in script.up_statements:
            act, tgt, click, sv = UPIntentClassifier.classify(up.raw)
            up.action, up.target_element = act, tgt
            up.click_element, up.state_value = click, sv

        print(f"\n{'─'*65}")
        print(f"  {script.name}  |  UP={len(script.up_statements)}"
              f"  UM={len(script.um_statements)}  PDL={len(script.pdl_lines)}")
        print(f"{'─'*65}")
        for i, up in enumerate(script.up_statements):
            print(f"  UP[{i}]  {up.action.name:<25}  {up.raw[:58]}")
            if up.target_element:
                print(f"         target  → {up.target_element}")
            if up.state_value:
                print(f"         state   → {up.state_value}")
            if up.click_element:
                print(f"         click   → {up.click_element}")
        return script

    def ocr_image(self, image_path: str) -> RefImageInfo:
        """OCR a single image and print all detected regions."""
        info = self._ocr_engine.process(image_path, Path(image_path).stem)
        print(f"\n{info.label}  ({info.width}x{info.height})  "
              f"{len(info.ocr_hits)} regions")
        for h in sorted(info.ocr_hits, key=lambda h: h.y1):
            print(f"  [{h.x1:4d},{h.y1:4d},{h.x2:4d},{h.y2:4d}] "
                  f"{h.conf:.2f}  \"{h.text}\"")
        return info


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse, sys

    ap = argparse.ArgumentParser(
        description="Generate pytest scripts from PDL+UP/UM test scripts."
    )
    ap.add_argument("script", help="Source .txt test script")
    ap.add_argument("images", nargs="*",
                    help="Reference images as 'path:label'  e.g. 3.png:cabin_lighting")
    ap.add_argument("--output",    default=None)
    ap.add_argument("--ref-dir",   default="./ref/CIL_/")
    ap.add_argument("--vlm",       action="store_true")
    ap.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL)
    ap.add_argument("--vlm-host",  default=DEFAULT_VLM_HOST)
    ap.add_argument("--summary",   action="store_true")
    args = ap.parse_args()

    lib = TestScriptGeneratorLibrary(use_vlm=args.vlm,
                                     vlm_model=args.vlm_model,
                                     vlm_host=args.vlm_host)
    if args.summary:
        lib.parse_and_summarise(args.script)
        sys.exit(0)

    ref_imgs = []
    for tok in args.images:
        if ":" in tok:
            p, lbl = tok.split(":", 1)
        else:
            p, lbl = tok, Path(tok).stem
        ref_imgs.append((p, lbl))

    lib.generate(script_path=args.script, reference_images=ref_imgs,
                 output_path=args.output, ref_output_dir=args.ref_dir)
