import json
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from analyze_music import init_yamnet, yamnet_analyze, tempo_analyze
from download_from_youtube import download_audio, download_comments, get_video_metadata
from config import openai_apikey

client = OpenAI(api_key=openai_apikey)

def generate_prompt(comments, instruments=None, genres=None, tempo=None, metadata=None):
    prompt = (
        "You are a professional music analyst. Based on the following YouTube comments, "
        "infer the characteristics of this song and respond in JSON format with the following fields.\n"
        "If a field is not mentioned in the comments, leave it as an empty string (do not guess).\n"
        "⚠️ Important: except for the 'title' and 'artist', all values must be written in **English**.\n"
        "⚠️ Important: Please return only the JSON object, without any explanation or markdown formatting (e.g. no ```json)\n\n"
        "Output format:\n"
        "{\n"
        "  \"genre\": \"\",\n"
        "  \"mood\": \"\",\n"
        "  \"instrument\": \"\",\n"
        "  \"theme\": \"\",\n"
        "  \"occasion\": \"\",\n"
        "  \"language\": \"\",\n"
        "  \"era\": \"\",\n"
        "  \"audience\": \"\",\n"
        "  \"tempo\": \"\",\n"
        "  \"title\": \"\",\n"
        "  \"artist\": \"\",\n"
        "  \"summary\": \"\"\n"
        "}\n"
    )

    # 加入影片標題與音訊提示
    info = f"The YouTube video title is: {metadata['title']}\n"
    info += "The following information was automatically extracted from the audio (for your reference):\n"

    if instruments:
        info += f"\nPossible instruments detected: {', '.join(instruments)}.\n"
    if genres:
        info += f"\nPossible genres suggested: {', '.join(genres)}.\n"
    if tempo:
        info += f"\nEstimated tempo: {tempo} BPM.\n"

    prompt = info + "\n" + prompt

    # 加入留言內容
    prompt += "\nUser comments and likes:\n"
    for i, comment in enumerate(comments, 1):
        prompt += f"{i}. {comment}\n"
    return prompt

def analyze_comments(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )

    text = response.choices[0].message.content.strip()

    # 嘗試把輸出轉為 JSON 格式
    try:
        # 如果開頭不是 `{`，先找出 JSON 區塊再轉換
        if not text.startswith("{"):
            json_start = text.find("{")
            json_str = text[json_start:]
        else:
            json_str = text

        return json_str
    except Exception as e:
        print("ChatGPT 輸出不是有效 JSON，原始內容如下：\n", text)
        raise e

def SummarizeMusic(url, id):
    metadata = get_video_metadata(url)
    download_audio(url, id)
    print("音檔下載完成")

    instruments, genres = yamnet_analyze(f"audio/{id}.mp3")
    tempo = tempo_analyze(f"audio/{id}.mp3")
    print("音訊分析完成")

    comments = download_comments(url, max_comments=200)
    print("留言下載完成")

    prompt = generate_prompt(comments, instruments=instruments, genres=genres, tempo=tempo, metadata=metadata)
    result = analyze_comments(prompt)

    return result

if __name__ == "__main__":
    print("啟動系統：初始化模型")
    init_yamnet()
    id = "test"
    url = input("請輸入YouTube網址：")
    metadata = get_video_metadata(url)
    
    print("正在下載音檔...")
    download_audio(url, id)
    print("音檔下載完成")

    print("正在進行音訊分析...")
    instruments, genres = yamnet_analyze("output.mp3")
    tempo = tempo_analyze("output.mp3")
    print("音訊分析完成")
    print("樂器：", instruments)
    print("曲風：", genres)
    print("節奏：", tempo)

    print("正在下載留言...")
    download_comments(url)
    print("留言下載完成")

    # 原始留言前處理
    all_comments = load_comments("comments.json", max_comments=200)

    # 透過 Embedding 篩選代表性留言
    selected_comments = select_representative_comments(all_comments, top_k=30)

    # 傳送給 GPT 分析
    prompt = generate_prompt(selected_comments, instruments=instruments, genres=genres, tempo=tempo, metadata=metadata)
    result = analyze_comments(prompt)
    
