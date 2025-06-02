import yt_dlp
from youtube_comment_downloader import YoutubeCommentDownloader

def download_audio(youtube_url, id):
    ydl_opts = {
        'format': 'bestaudio/best',
        'ffmpeg_location': 'C:/Program Files/ffmpeg/bin/ffmpeg.exe',
        'outtmpl': f'audio/{id}',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

def parse_like_count(text):
    """
    將「33萬」「1.2千」「89」等中文數字格式轉為整數。
    """
    text = text.strip()
    if isinstance(text, int):
        return text
    if not text:
        return 0

    multiplier = 1
    if '萬' in text:
        multiplier = 10000
        text = text.replace('萬', '')
    elif '千' in text:
        multiplier = 1000
        text = text.replace('千', '')

    try:
        return int(float(text) * multiplier)
    except ValueError:
        return 0

def download_comments(youtube_url, max_comments=500, min_likes=1, min_length=10):
    downloader = YoutubeCommentDownloader()
    comments_data = []

    for comment in downloader.get_comments_from_url(youtube_url, sort_by=0):
        if len(comments_data) >= max_comments:
            break

        text = comment.get("text", "").strip()
        likes = comment.get("votes", 0)

        if len(text) >= min_length and parse_like_count(likes) >= min_likes:
            comment_info = {
                "text": text,
                "likes": likes,
            }
            comments_data.append(comment_info)

    return comments_data

def get_video_metadata(youtube_url):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        metadata = {
            "title": info.get("title"),
            "description": info.get("description"),
            "uploader": info.get("uploader"),
            "upload_date": info.get("upload_date"),
            "tags": info.get("tags"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "duration": info.get("duration"),
            "categories": info.get("categories"),
        }
        return metadata

def extract_playlist_urls(playlist_url):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'skip_download': True
    }

    videos = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        entries = info.get('entries', [])

        for entry in entries:
            videos.append({
                "id": entry.get('id'),
                "title": entry.get("title"),
                "url": f"https://www.youtube.com/watch?v={entry.get('id')}",
            })

    return videos

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=kJQP7kiw5Fk"
    # metadata = get_video_metadata(url)

    # print("標題：", metadata["title"])
    # print("上傳時間：", metadata["upload_date"])
    # print("標籤：", metadata["tags"])
    # print("觀看數：", metadata["view_count"])

    # print("正在下載音檔...")
    # download_audio(url, "test")
    # print("音檔下載完成！")

    comments = download_comments(url, max_comments=100)
    print(comments)

    # url = "https://www.youtube.com/playlist?list=PLA9x9-eADvOq0BC1xUWADX4JmQ2c3dcFm"
    # videos = extract_playlist_urls(url)
    # for i in videos:
    #     print(i)