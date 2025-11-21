import streamlit as st
import pandas as pd
import json
import streamlit.components.v1 as components
from datasets import load_dataset

st.set_page_config(page_title="Video QA Error Analysis", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    h1 { margin-bottom: 0px; }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Visualize videos & Error Analysis")

@st.cache_resource
def get_dataset():
    return load_dataset("petkopetkov/EgoTextVQA")

with st.spinner("Loading HF Dataset..."):
    try:
        splits = get_dataset()
        data = splits["train"] 
        df = pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

def fix_hf_url(url):
    if url and "tree/main" in url:
        return url.replace("tree/main", "resolve/main")
    return url

if 'video_url' in df.columns:
    df['video_url'] = df['video_url'].apply(fix_hf_url)

st.sidebar.header("Control Panel")

st.sidebar.subheader("1. Load Predictions")
pred_file = st.sidebar.file_uploader("Upload predictions.json", type=["json"])
model_key = st.sidebar.text_input("Model Key in JSON", value="gemini-2.5-flash")

predictions_map = {} 

if pred_file:
    try:
        pred_data = json.load(pred_file)
        for item in pred_data:
            v_id = str(item.get('video_id'))
            q_text = item.get('question')
            predictions_map[(v_id, q_text)] = item
        
        st.sidebar.success(f"Loaded {len(pred_data)} predictions!")
    except Exception as e:
        st.sidebar.error(f"Error reading json: {e}")

st.sidebar.subheader("2. Select Sample")

if predictions_map:
    pred_vid_ids = set([k[0] for k in predictions_map.keys()])
    df_vid_ids = set(df['video_id'].astype(str).unique())
    common_ids = list(pred_vid_ids.intersection(df_vid_ids))
    
    if common_ids:
        unique_videos = sorted(common_ids)
    else:
        st.warning("No matching Video IDs found between Dataset and Predictions.")
        unique_videos = df['video_id'].unique()
else:
    unique_videos = df['video_id'].unique()

selected_video_id = st.sidebar.selectbox("Select Video ID", unique_videos)

video_subset = df[df['video_id'] == selected_video_id].sort_values(by='timestamp')

if video_subset.empty:
    st.error("No data found for this video.")
    st.stop()

video_url = video_subset.iloc[0]['video_url']

qa_timeline = []
for _, row in video_subset.iterrows():
    vid_str = str(row['video_id'])
    q_str = row['question']
    pred_item = predictions_map.get((vid_str, q_str), None)
    
    model_pred = "N/A"
    is_correct = "unknown"
    score = 0
    
    if pred_item:
        model_pred = pred_item.get(model_key, "N/A")
        acc_raw = str(pred_item.get('acc', 'no')).lower()
        is_correct = "yes" if "yes" in acc_raw else "no"
        score = pred_item.get('score', 0)

    qa_timeline.append({
        "timestamp": float(row['timestamp']),
        "question": row['question'],
        "answer": row['answer'] if isinstance(row['answer'], str) else str(row['answer']),
        "type": row.get('question_type', 'Unknown'),
        "prediction": model_pred,
        "is_correct": is_correct,
        "score": score
    })

qa_json = json.dumps(qa_timeline)

html_code = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    body {{
        margin: 0;
        background-color: #0e1117;
        font-family: "Source Sans Pro", sans-serif;
        color: white;
        overflow: hidden;
    }}
    
    .video-wrapper {{
        position: relative;
        width: 100%;
        height: 75vh;
        background: black;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }}

    video {{
        width: 100%;
        height: 100%;
        object-fit: contain;
    }}

    .overlay {{
        position: absolute;
        bottom: 30px; 
        left: 50%;
        transform: translateX(-50%);
        
        width: 85%;
        max-width: 800px;
        
        max-height: 55vh; 
        
        background: rgba(15, 15, 15, 0.95);
        backdrop-filter: blur(8px);
        padding: 15px 20px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        opacity: 0;
        transition: opacity 0.3s ease, transform 0.3s ease;
        pointer-events: none; 
        display: flex;
        flex-direction: column;
        gap: 10px;
    }}

    .overlay.active {{ opacity: 1; }}
    .overlay.user-hidden {{ opacity: 0 !important; pointer-events: none !important; }}
    
    .overlay.active > * {{ pointer-events: auto; }}

    .toggle-btn {{
        position: absolute;
        top: 15px;
        left: 15px;
        background: rgba(0, 0, 0, 0.6);
        color: white;
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 4px;
        padding: 6px 12px;
        cursor: pointer;
        font-size: 0.8rem;
        font-weight: bold;
        z-index: 10;
        transition: background 0.2s;
        font-family: sans-serif;
    }}
    .toggle-btn:hover {{ background: rgba(255, 255, 255, 0.2); }}

    .section-title {{
        font-size: 0.7rem;
        text-transform: uppercase;
        color: #888;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }}

    .question-text {{
        font-size: 1.1rem;
        font-weight: 600;
        color: #fff;
        line-height: 1.3;
        max-height: 60px; 
        overflow-y: auto;
    }}

    .comparison-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
        flex-grow: 1;
        min-height: 0; 
    }}

    .box {{
        background: rgba(255, 255, 255, 0.05);
        padding: 10px 12px;
        border-radius: 8px;
        border-left: 3px solid #555;
        display: flex;
        flex-direction: column;
        max-height: 220px; 
    }}

    .box.gt {{ border-left-color: #4caf50; }}
    .box.pred.correct {{ border-left-color: #4caf50; background: rgba(76, 175, 80, 0.1); }}
    .box.pred.incorrect {{ border-left-color: #ff4b4b; background: rgba(255, 75, 75, 0.15); }}
    .box.pred.unknown {{ border-left-color: #888; }}

    .box-label {{
        font-size: 0.7rem;
        font-weight: bold;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        flex-shrink: 0;
    }}
    
    .text-content {{
        font-size: 0.9rem;
        color: #e0e0e0;
        line-height: 1.4;
        overflow-y: auto;
        flex-grow: 1;
        padding-right: 6px;
        word-wrap: break-word;
        white-space: pre-wrap;
        scrollbar-width: thin;
        scrollbar-color: rgba(255,255,255,0.3) transparent;
    }}
    
    .text-content::-webkit-scrollbar {{ width: 4px; }}
    .text-content::-webkit-scrollbar-track {{ background: transparent; }}
    .text-content::-webkit-scrollbar-thumb {{ background-color: rgba(255,255,255,0.3); border-radius: 10px; }}

    .score-badge {{
        background: rgba(0,0,0,0.5);
        padding: 1px 6px;
        border-radius: 3px;
        font-family: monospace;
        font-size: 0.75rem;
    }}

    .timestamp-indicator {{
        position: absolute;
        top: 15px;
        right: 15px;
        font-family: monospace;
        background: rgba(0,0,0,0.6);
        color: #00d4ff;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.9rem;
        font-weight: bold;
        pointer-events: none;
    }}

</style>
</head>
<body>

<div class="video-wrapper">
    <video id="mainVideo" controls autoplay>
        <source src="{video_url}" type="video/mp4">
        Your browser does not support the video tag.
    </video>

    <button id="toggleOverlayBtn" class="toggle-btn">👁️ Hide Overlay</button>

    <div id="overlay" class="overlay">
        <div>
            <div class="section-title">Question</div>
            <div id="qText" class="question-text">...</div>
        </div>

        <div class="comparison-grid">
            
            <div class="box gt">
                <div class="box-label" style="color: #4caf50;">GROUND TRUTH</div>
                <div id="aText" class="text-content">...</div>
            </div>

            <div id="predBox" class="box pred unknown">
                <div class="box-label" id="predLabelColor">
                    <span>MODEL PREDICTION</span>
                    <span id="scoreText" class="score-badge">Score: -</span>
                </div>
                <div id="pText" class="text-content">...</div>
            </div>
        </div>
    </div>
    
    <div id="timeDebug" class="timestamp-indicator">0.0s</div>
</div>

<script>
    const qaData = {qa_json};
    
    const video = document.getElementById('mainVideo');
    const overlay = document.getElementById('overlay');
    const toggleBtn = document.getElementById('toggleOverlayBtn');
    
    const qText = document.getElementById('qText');
    const aText = document.getElementById('aText');
    const pText = document.getElementById('pText');
    const scoreText = document.getElementById('scoreText');
    const predBox = document.getElementById('predBox');
    const predLabelColor = document.getElementById('predLabelColor');
    const timeDebug = document.getElementById('timeDebug');

    let currentActiveIdx = -1;
    let isHiddenByUser = false;

    // Toggle Logic
    toggleBtn.onclick = () => {{
        isHiddenByUser = !isHiddenByUser;
        if (isHiddenByUser) {{
            overlay.classList.add('user-hidden');
            toggleBtn.innerText = "👁️ Show Overlay";
        }} else {{
            overlay.classList.remove('user-hidden');
            toggleBtn.innerText = "👁️ Hide Overlay";
        }}
    }};

    video.ontimeupdate = function() {{
        const t = video.currentTime;
        timeDebug.innerText = t.toFixed(1) + "s";

        let activeIdx = -1;
        for (let i = 0; i < qaData.length; i++) {{
            if (t >= qaData[i].timestamp) {{
                activeIdx = i;
            }} else {{
                break;
            }}
        }}

        if (activeIdx !== currentActiveIdx) {{
            currentActiveIdx = activeIdx;

            if (activeIdx > -1) {{
                const item = qaData[activeIdx];
                qText.innerText = item.question;
                aText.innerText = item.answer;
                pText.innerText = item.prediction;
                scoreText.innerText = "Score: " + item.score;

                predBox.classList.remove('correct', 'incorrect', 'unknown');
                if (item.prediction === "N/A") {{
                    predBox.classList.add('unknown');
                    predLabelColor.style.color = "#888";
                }} else if (item.is_correct === "yes") {{
                    predBox.classList.add('correct');
                    predLabelColor.style.color = "#4caf50";
                }} else {{
                    predBox.classList.add('incorrect');
                    predLabelColor.style.color = "#ff4b4b";
                }}
                
                overlay.classList.add('active');
            }} else {{
                overlay.classList.remove('active');
            }}
        }}
    }};
</script>
</body>
</html>
"""

components.html(html_code, height=750)

st.divider()
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 📋 Timeline Analysis")
    
    display_data = []
    for item in qa_timeline:
        display_data.append({
            "Time": item['timestamp'],
            "Question": item['question'],
            "Ground Truth": item['answer'],
            "Prediction": item['prediction'],
            "Acc": "✅" if item['is_correct'] == "yes" else "❌" if item['is_correct'] == "no" else "❓",
            "Score": item['score']
        })
    
    st.dataframe(
        pd.DataFrame(display_data), 
        use_container_width=True,
        column_config={
            "Time": st.column_config.NumberColumn("Time", format="%.1fs"),
            "Question": st.column_config.TextColumn("Question", width="medium"),
            "Prediction": st.column_config.TextColumn("Prediction", width="large"),
        },
        hide_index=True
    )

with col2:
    if pred_file:
        st.markdown("### Video Metrics")
        df_metrics = pd.DataFrame(qa_timeline)
        if not df_metrics.empty:
            scored_items = df_metrics[df_metrics['prediction'] != "N/A"]
            if not scored_items.empty:
                total = len(scored_items)
                correct = len(scored_items[scored_items['is_correct'] == 'yes'])
                avg_score = scored_items['score'].mean()
                st.metric("Accuracy", f"{correct}/{total}", f"{(correct/total)*100:.1f}%")
                st.metric("Avg Score", f"{avg_score:.2f}")
            else:
                st.info("No predictions found for this specific video.")