import yt_dlp
from youtube_comment_downloader import YoutubeCommentDownloader
import json

def download_audio(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'ffmpeg_location': 'C:/Program Files/ffmpeg/bin/ffmpeg.exe',
        'outtmpl': 'output.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

def download_comments(youtube_url):
    downloader = YoutubeCommentDownloader()
    comments_data = []

    for comment in downloader.get_comments_from_url(youtube_url, sort_by=0):
        comment_info = {
            "text": comment.get("text", ""),
            "likes": comment.get("votes", 0),
            "time": comment.get("time", "")
        }
        comments_data.append(comment_info)

    with open("comments.json", "w", encoding="utf-8") as f:
        json.dump(comments_data, f, ensure_ascii=False, indent=2)