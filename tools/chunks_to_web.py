# -*- coding: utf-8 -*-
"""
색인앱(app.py)이 만든  indexes/<세트>/chunks.jsonl  →  웹 데모용  web/public/chunks.json  변환.

사용:
  python tools/chunks_to_web.py                      # 사용 가능한 세트 목록
  python tools/chunks_to_web.py <세트이름>            # 변환 → web/public/chunks.json (기본)
  python tools/chunks_to_web.py <세트이름> <출력.json> # 다른 경로로 출력(미리보기 등)

⚠️ 변환 결과를 공개 배포(GitHub Pages)하면 그 매뉴얼 내용이 인터넷에 공개됩니다.
   공개 가능한 자료만 web/public/chunks.json 으로 넣으세요(보안 매뉴얼 금지).
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # tools/ 의 상위 = 리포 루트
INDEXES = ROOT / "indexes"
DEFAULT_OUT = ROOT / "web" / "public" / "chunks.json"


def list_sets():
    if not INDEXES.exists():
        return []
    return sorted(
        d.name for d in INDEXES.iterdir()
        if d.is_dir() and (d / "chunks.jsonl").exists()
    )


def convert(set_name: str, out_path: Path) -> int:
    jsonl = INDEXES / set_name / "chunks.jsonl"
    if not jsonl.exists():
        sys.exit(
            f"[X] 파일 없음: {jsonl}\n"
            f"    먼저 app.py '색인 생성' 탭에서 '{set_name}' 세트를 색인하세요"
            f"(indexes/{set_name}/chunks.jsonl 이 생성됩니다)."
        )
    records = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            text = (c.get("text") or "").strip()
            if not text:
                continue
            records.append({
                "id": c.get("id"),
                "text": text,
                "source": c.get("source_file") or c.get("source") or set_name,
                "page": c.get("page"),
                "section": c.get("section") or "[검증 필요]",
            })
    if not records:
        sys.exit(f"[X] 변환할 청크가 없습니다: {jsonl}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)

    print(f"[OK] {len(records)}개 청크 변환 -> {out_path}")
    print("[!] 공개 배포 시 이 내용이 인터넷에 노출됩니다. 공개 가능한 자료인지 확인 후 배포하세요.")
    print("    재배포:  cd web && npm run build   그리고  git add web/public/chunks.json && git commit && git push")
    return len(records)


def main():
    args = sys.argv[1:]
    if not args:
        sets = list_sets()
        if sets:
            print("사용 가능한 세트:", ", ".join(sets))
            print("변환:  python tools/chunks_to_web.py <세트이름>")
        else:
            print("색인된 세트(chunks.jsonl)가 없습니다. app.py '색인 생성' 탭에서 먼저 색인하세요.")
        return
    set_name = args[0]
    out_path = Path(args[1]) if len(args) > 1 else DEFAULT_OUT
    convert(set_name, out_path)


if __name__ == "__main__":
    main()
