from summarize_music import SummarizeMusic
from analyze_music import init_yamnet
from download_from_youtube import extract_playlist_urls
import csv
import time
import argparse
import json

def batch_process_csv(videos, output_csv, limit=None):
    with open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:

        if limit:
            videos = videos[:limit]

        fieldnames = ["id", "title", "artist", "genre", "mood", "instrument", "theme",
                      "occasion", "language", "era", "audience", "tempo", "summary", "youtube_url"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, video in enumerate(videos):
            try:
                url = video["url"]
                id = video["id"]
                title = video["title"]

                if not url:
                    print(f"id: {id} 沒有有效的 YouTube 連結，跳過。")
                    continue
                
                print(f"\n第 {i} 首處理中：{id}")
                result_json_str = SummarizeMusic(url, id)

                result = json.loads(result_json_str)
                writer.writerow({
                    "id": id,
                    "title": result.get("title", title),
                    "artist": result.get("artist", ""),
                    "genre": result.get("genre", ""),
                    "mood": result.get("mood", ""),
                    "instrument": result.get("instrument", ""),
                    "theme": result.get("theme", ""),
                    "occasion": result.get("occasion", ""),
                    "language": result.get("language", ""),
                    "era": result.get("era", ""),
                    "audience": result.get("audience", ""),
                    "tempo": result.get("tempo", ""),
                    "summary": result.get("summary", ""),
                    "youtube_url": url
                })
            except Exception as e:
                print(f"id: {id} 發生錯誤：{e}")
                continue
            time.sleep(2)  # 控制 API 請求速度

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch YouTube song analyzer")
    parser.add_argument("--limit", type=int, default=None, help="只分析前 N 首歌曲（預設為全部）")
    args = parser.parse_args()
    
    print("啟動系統：初始化模型")
    init_yamnet()

    playlist_url = "https://www.youtube.com/playlist?list=PL15B1E77BB5708555"
    videos = extract_playlist_urls(playlist_url)

    batch_process_csv(videos, "output.csv", limit=args.limit)