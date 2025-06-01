import json
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from analyze_music import analyze_instruments, analyze_tempo
from download_from_youtube import download_audio, download_comments
from config import openai_apikey

client = OpenAI(api_key=openai_apikey)

def load_comments(filepath="comments.json", max_comments=50, min_length=10):
    with open(filepath, "r", encoding="utf-8") as f:
        comments = json.load(f)

    # 過濾條件：
    # 1. 有文字
    # 2. 按讚數（可為 0）
    # 3. 文字長度 >= min_length
    filtered = [
        c for c in comments
        if c.get("text") and len(c["text"].strip()) >= min_length
    ]

    # 依據按讚數排序，從高到低
    sorted_comments = sorted(filtered, key=lambda x: x.get("likes", 0), reverse=True)

    # 取前 N 筆留言文字
    return [c["text"] for c in sorted_comments[:max_comments]]

def get_embeddings(texts):
    embeddings = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        embeddings.extend([d.embedding for d in response.data])
    return np.array(embeddings)

def select_representative_comments(comments, top_k=20):
    print("正在計算留言 Embedding...")

    embeddings = get_embeddings(comments)

    # 計算中心點（平均向量）
    center = np.mean(embeddings, axis=0, keepdims=True)

    # 計算每個留言與中心的餘弦距離
    distances = cosine_distances(embeddings, center).flatten()

    # 按距離排序，距離越小代表越接近中心（越代表主流語意）
    sorted_indices = np.argsort(distances)

    selected_comments = [comments[i] for i in sorted_indices[:top_k]]
    return selected_comments

def generate_prompt(comments, instruments=None, tempo=None):
    prompt = (
        "你是一位專業的音樂評論分析師，請根據下列留言，推斷這首歌曲的各種特徵，並以 JSON 格式輸出以下欄位。\n"
        "如果某個欄位的資訊在留言中沒有被提到，請保留該欄位為空字串（不要憑空猜測）。\n"
        "\n"
        "- \"genre\"：歌曲的類型或風格（如 Pop、Rock、Jazz 等）。\n"
        "- \"mood\"：歌曲傳達的情緒（如 感動、悲傷、快樂、平靜 等）。\n"
        "- \"instrument\"：歌曲中使用的主要樂器（如 吉他、鋼琴、鼓 等）。\n"
        "- \"theme\"：歌曲傳達的主題（如 愛情、自由、回憶、青春 等）。\n"
        "- \"occasion\"：適合聆聽這首歌的情境（如 派對、夜晚開車、運動、療癒 等）。\n"
        "- \"language\"：歌曲使用的語言（如 中文、英文、西班牙文 等）。\n"
        "- \"era\"：歌曲的年代感或風格年代（如 1980s、2000s、現代 等）。\n"
        "- \"audience\"：這首歌可能吸引的聽眾族群（如 青少年、情侶、大眾 等）。\n"
        "- \"tempo\"：歌曲的節奏速度，以 BPM（每分鐘節拍數）為單位（如 120、90～100）。\n"
        "- \"time_signature\"：歌曲的拍號（如 4/4、3/4、6/8 等）。\n"
        "- \"title\"：歌曲名稱（如果留言中有提到）。\n"
        "- \"artist\"：歌曲的演唱者或創作者（如果留言中有提到）。\n"
        "- \"summary\"：請總結這首歌在留言中的整體感受與特徵。\n"
    )

    info = ""

    # 加入 Demucs 樂器識別補充
    if instruments:
        instr_line = "以下是自動從音訊分析得到的樂器資訊，供你參考：\n"
        instr_line += f"音訊中出現的樂器包括：{'、'.join(instruments)}。\n"
        info = info + "\n" + instr_line

    if tempo:
        info_tempo = f"以下是自動從音訊分析得到的節奏資訊，供你參考：{tempo} bpm\n"
        info = info + "\n" + info_tempo

    prompt = info + "\n" + prompt

    prompt += "\n留言如下：\n"
    for i, comment in enumerate(comments, 1):
        prompt += f"{i}. {comment}\n"
    return prompt

def analyze_comments(comments, instruments, tempo):
    prompt = generate_prompt(comments, instruments, tempo)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    url = input("請輸入YouTube網址：")
    
    print("正在下載音檔...")
    download_audio(url)
    print("音檔下載完成！")

    print("正在進行音源分離...")
    instruments = analyze_instruments("output.mp3")
    print("樂器分析完成：", instruments)

    print("正在進行節奏分析...")
    tempo = analyze_tempo("output.mp3")
    print("節奏分析完成：", tempo)

    print("正在下載留言...")
    download_comments(url)
    print("留言下載完成！")

    # 原始留言前處理
    all_comments = load_comments("comments.json", max_comments=200)

    # 透過 Embedding 篩選代表性留言
    selected_comments = select_representative_comments(all_comments, top_k=30)

    # 傳送給 GPT 分析
    result = analyze_comments(selected_comments, instruments=instruments, tempo=tempo)
    print("分析結果：\n", result)
