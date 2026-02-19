# Set environment variable to avoid PaddleOCR network check hang
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import streamlit as st
import time
from PIL import Image
from vlm_engine import VLMEngine
import fitz  # PyMuPDF
import numpy as np
import io
import json
import concurrent.futures
import datetime
import traceback
import threading
import re
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# Page Config
st.set_page_config(page_title="Math Problem Extractor", page_icon="📝", layout="wide")

def run_with_ctx(ctx, fn, *args, **kwargs):
    """Helper to inject Streamlit context into worker threads."""
    if ctx:
        add_script_run_ctx(ctx)
    return fn(*args, **kwargs)

def render_single_problem(idx, prob, display_image, vlm_engine):
    """Renders a single problem card."""
    # Problem Title and Score
    score_info = f" ({prob.get('score')} pts)" if prob.get('score') else ""
    st.markdown(f"**[Problem {prob.get('number', idx+1)}]{score_info}**")
    
    # Prioritize Unified Content (Full LaTeX)
    if prob.get("content"):
        st.markdown(prob.get("content"))
    else:
        # Backward compatibility / Fallback for separate fields
        content = prob.get("text", "")
        st.markdown(content)
        
        # Handle multiple formulas
        formulas = prob.get("formulas", [])
        if not formulas and prob.get("formula"):
            formulas = [prob.get("formula")]
            
        for f in formulas:
            if f and f.strip() and f not in content:
                if not f.startswith("$$"):
                    f_display = f"$${f}$$"
                else:
                    f_display = f
                st.markdown(f"**Formula:** {f_display}")
    
    # Difficulty Level Display
    if prob.get("difficulty"):
        diff = prob.get("difficulty")
        level = int(diff.get("level", 0))
        stars = "⭐" * level + "☆" * (6 - level)
        st.markdown(f"**Difficulty:** {stars} (Level {level}/6)")
        
        with st.expander("📊 Difficulty Analysis Details"):
            st.markdown(f"- **Conceptual:** {diff.get('conceptual', 'N/A')}")
            st.markdown(f"- **Logical:** {diff.get('logical', 'N/A')}")
            st.markdown(f"- **Computational:** {diff.get('computational', 'N/A')}")
            st.info(f"Summary: {diff.get('summary', '')}")

    # Solving Process (Visible by default)
    if prob.get("solution"):
        st.markdown("**📝 Solving Process:**")
        st.markdown(prob.get("solution"))
    
    # Options
    if prob.get("options"):
        st.markdown("**Options:**")
        for opt_idx, opt in enumerate(prob.get("options", [])):
            st.markdown(f"> {opt_idx+1}. {opt}")

    # Graph Section
    desc = prob.get("image_description", "")
    if desc and desc.lower() != "none" and len(desc) > 5 and display_image is not None:
        st.info(f"🔍 Visual Element: {desc}")
        coords = vlm_engine.detect_graph_coordinates(display_image, desc)
        if coords:
            w, h = display_image.size
            ymin, xmin, ymax, xmax = coords
            
            # Add 5% margin to prevent cutoff
            margin_x = (xmax - xmin) * 0.05
            margin_y = (ymax - ymin) * 0.05
            
            left = max(0, (xmin - margin_x) * w / 1000)
            top = max(0, (ymin - margin_y) * h / 1000)
            right = min(w, (xmax + margin_x) * w / 1000)
            bottom = min(h, (ymax + margin_y) * h / 1000)
            
            graph_crop = display_image.crop((left, top, right, bottom))
            st.image(graph_crop, width=400)
    
    # Raw Data for Visibility
    if prob.get("raw_content"):
        with st.expander("📄 Raw Extraction Data"):
            st.code(prob.get("raw_content"))

    st.write("---")

def save_to_library(result_json, source_name):
    """Saves problems to a structured directory."""
    base_dir = "problem_library"
    safe_source = "".join([c for c in source_name if c.isalnum() or c in (' ', '.', '_')]).strip()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    source_folder = f"{safe_source}_{timestamp}"
    
    saved_count = 0
    for idx, prob in enumerate(result_json.get("problems", [])):
        level = prob.get("difficulty", {}).get("level", 0)
        level_dir = os.path.join(base_dir, f"Level_{level}", source_folder)
        os.makedirs(level_dir, exist_ok=True)
        file_path = os.path.join(level_dir, f"Problem_{prob.get('number', idx+1)}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(prob, f, ensure_ascii=False, indent=4)
        saved_count += 1
    return saved_count

def library_browser_tab(vlm_engine):
    """UI for browsing saved problems."""
    st.header("📚 Problem Library")
    base_dir = "problem_library"
    if not os.path.exists(base_dir):
        st.info("No problems saved yet.")
        return
    levels = sorted([d for d in os.listdir(base_dir) if d.startswith("Level_")])
    if not levels:
        st.info("Library is empty.")
        return
    selected_level = st.selectbox("Select Difficulty Level", levels)
    level_path = os.path.join(base_dir, selected_level)
    sources = sorted(os.listdir(level_path)) if os.path.exists(level_path) else []
    if not sources:
        st.write("No sources found.")
        return
    selected_source = st.selectbox("Select Source", sources)
    source_path = os.path.join(level_path, selected_source)
    probs = sorted([f for f in os.listdir(source_path) if f.endswith(".json")])
    st.divider()
    
    selected_prob_file = st.selectbox("Select Problem", probs)
    if selected_prob_file:
        with open(os.path.join(source_path, selected_prob_file), "r", encoding="utf-8") as f:
            prob_data = json.load(f)
            render_single_problem(0, prob_data, None, vlm_engine)
            
            # --- AI Chat Interaction ---
            st.markdown("### 💬 Chat with AI about this Problem")
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if prompt := st.chat_input("Ask a question about this problem..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("AI is thinking..."):
                        context_prompt = (
                            f"The user is asking about the following math problem:\n\n"
                            f"Problem: {prob_data.get('text')}\n"
                            f"Solution: {prob_data.get('solution')}\n"
                            f"Options: {prob_data.get('options')}\n\n"
                            f"User Query: {prompt}"
                        )
                        response = vlm_engine.solve_problem(context_prompt)
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})

def display_results(result_json, display_image, vlm_engine):
    """
    Displays results: LaTeX-rendered problems only.
    """
    # st.subheader("1. Extraction Result (JSON)")
    # st.json(result_json)
    
    # st.divider()
    
    st.subheader("Rendered View (LaTeX)")
    if "problems" not in result_json or not result_json["problems"]:
        st.warning("No problems were found.")
        return

    for idx, prob in enumerate(result_json["problems"]):
        render_single_problem(idx, prob, display_image, vlm_engine)


# Helper Functions

def parse_problems_xml(text):
    """Helper to extract problems from XML-like output."""
    problems = []
    # Find all <problem> blocks
    prob_blocks = re.findall(r'<problem>(.*?)</problem>', text, re.DOTALL)
    
    for block in prob_blocks:
        p = {}
        # Store original block for full visibility
        p["raw_content"] = block.strip()
        
        # Extract fields using regex
        num_match = re.search(r'<number>(.*?)</number>', block, re.DOTALL)
        num_str = num_match.group(1).strip() if num_match else str(len(problems) + 1)
        try:
            p["number"] = int(num_str)
        except ValueError:
            p["number"] = num_str
        
        text_match = re.search(r'<text>(.*?)</text>', block, re.DOTALL)
        p["text"] = text_match.group(1).strip() if text_match else ""
        
        # Unified Content (LaTeX integrated)
        content_match = re.search(r'<content>(.*?)</content>', block, re.DOTALL)
        p["content"] = content_match.group(1).strip() if content_match else ""
        
        # Support multiple formulas (Legacy/Metadata)
        formulas = re.findall(r'<formula>(.*?)</formula>', block, re.DOTALL)
        p["formulas"] = [f.strip() for f in formulas]
        # Keep singular for backward compatibility/older saved files
        p["formula"] = p["formulas"][0] if p["formulas"] else ""
        
        opts_match = re.search(r'<options>(.*?)</options>', block, re.DOTALL)
        if opts_match:
            opts_content = opts_match.group(1)
            p["options"] = [opt.strip() for opt in re.findall(r'<option>(.*?)</option>', opts_content, re.DOTALL)]
        else:
            p["options"] = []

        score_match = re.search(r'<score>(.*?)</score>', block, re.DOTALL)
        p["score"] = score_match.group(1).strip() if score_match else ""

        img_match = re.search(r'<image_description>(.*?)</image_description>', block, re.DOTALL)
        p["image_description"] = img_match.group(1).strip() if img_match else ""
        
        problems.append(p)
    
    return {"problems": problems}

def parse_answer(solution_text):
    """Simple heuristic to find the final answer."""
    matches = re.finditer(r'(?:답|Answer|result)[:\s]+(.*?)(?:\n|$)', solution_text, re.IGNORECASE)
    for match in matches:
        cand = match.group(1).strip()
        num_match = re.search(r'[-+]?\d*\.?\d+', cand)
        if num_match:
            try: return float(num_match.group())
            except: continue
    
    last_lines = solution_text.split('\n')[-3:]
    for line in reversed(last_lines):
         num_match = re.search(r'[-+]?\d*\.?\d+', line)
         if num_match:
             try: return float(num_match.group())
             except: pass
    return None

def solve_task(prob, img, vlm_engine):
    """Worker function for parallel solving with smart retries."""
    def qextract(tag, text):
        m = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
        return m.group(1).strip() if m else ""

    prob_text = f"{prob.get('text', '')} {prob.get('formula', '')}"
    if prob.get('options'):
        prob_text += f"\nOptions: {', '.join(prob.get('options'))}"
    
    try:
        # Initial solve
        res = vlm_engine.solve_problem(prob_text, stream=False)
        full_sol = res.text if hasattr(res, 'text') else str(res)
        
        # Smart Retry Check
        ans_num = parse_answer(full_sol)
        needs_retry = False
        if ans_num is not None:
            if prob.get('options'):
                # Extract numbers from options
                opt_nums = []
                for opt in prob['options']:
                    oms = re.findall(r'[-+]?\d*\.?\d+', opt)
                    if oms:
                         try: opt_nums.append(float(oms[-1])) 
                         except: pass
                if not any(abs(o - ans_num) < 0.01 for o in opt_nums):
                    needs_retry = True
            elif abs(ans_num) > 1000:
                needs_retry = True

        if needs_retry:
            # Retry with image scan
            res_retry = vlm_engine.solve_problem(prob_text, stream=False, image=img)
            full_sol = res_retry.text if hasattr(res_retry, 'text') else str(res_retry)

        # Parse final result
        f_match = re.search(r'<final_answer>(.*?)</final_answer>', full_sol, re.DOTALL)
        a_match = re.search(r'<analysis>.*?</analysis>', full_sol, re.DOTALL)
        
        clean = full_sol
        if f_match: clean = clean.replace(f_match.group(0), "")
        if a_match: clean = clean.replace(a_match.group(0), "")
        
        diff = {}
        if a_match:
            ac = a_match.group(0)
            diff = {
                "level": int(qextract("level", ac) or 0),
                "conceptual": qextract("conceptual_reason", ac),
                "logical": qextract("logical_reason", ac),
                "computational": qextract("computational_reason", ac),
                "summary": qextract("summary", ac)
            }
        return {"solution": clean.strip(), "difficulty": diff}
    except Exception as e:
        return {"solution": f"Error: {e}", "difficulty": {}}

def run_hybrid_process(img, vlm_engine, status_container=None):
    """Parallel problem solving for a single image/page."""
    if not vlm_engine: return None
    
    # Use provided container or root st for updates
    ui = status_container if status_container else st
    
    # Stage 1: Extraction
    raw_result = vlm_engine.extract_math_problems(img)
    try:
        data = parse_problems_xml(raw_result)
        if "problems" in data and data["problems"]:
            ui.info(f"Solved {len(data['problems'])} problems in parallel.")
            # Individual problem status placeholders inside the page container
            status_placeholders = [ui.empty() for _ in data["problems"]]
            for i, p in enumerate(data["problems"]):
                status_placeholders[i].caption(f"Prob {p['number']}: 🔄 Solving...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                ctx = get_script_run_ctx()
                futures = {executor.submit(run_with_ctx, ctx, solve_task, p, img, vlm_engine): i for i, p in enumerate(data["problems"])}
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    try:
                        res = future.result()
                        data["problems"][i].update(res)
                        status_placeholders[i].caption(f"Prob {data['problems'][i]['number']}: ✅ Done")
                    except Exception as e:
                        status_placeholders[i].error(f"Prob {data['problems'][i]['number']}: ❌ Failed")
        return data
    except Exception as e:
        ui.error(f"Parsing Error: {e}")
        return None

# --- Sidebar Configuration ---

st.sidebar.title("Configuration")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))

if not gemini_api_key:
    st.sidebar.warning("Please provide a Gemini API Key.")
    vlm_engine = None
else:
    vlm_engine = VLMEngine(api_key=gemini_api_key)
    
    st.sidebar.divider()
    st.sidebar.markdown("### 📚 Reference Context Cache")
    
    if vlm_engine.reference_cache_name:
        st.sidebar.success(f"✅ Cache Active")
        st.sidebar.caption(f"ID: {vlm_engine.reference_cache_name[:15]}...")
    else:
        st.sidebar.warning("⚠️ No Active Cache")
        
    if st.sidebar.button("Update/Create Cache"):
        with st.sidebar.status("Updating Cache...", expanded=True) as status:
            try:
                import subprocess
                st.write("📤 Uploading books...")
                result = subprocess.run(
                    ["python", "upload_context.py"], 
                    capture_output=True, 
                    text=True, 
                    env={**os.environ, "GEMINI_API_KEY": gemini_api_key}
                )
                if result.returncode == 0:
                    st.write("✅ Upload complete!")
                    vlm_engine.load_cache_info()
                    status.update(label="Cache Updated!", state="complete", expanded=False)
                    st.rerun()
                else:
                    st.error("Failed to update cache")
                    st.code(result.stderr)
                    status.update(label="Update Failed", state="error")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.sidebar.markdown("### 📤 Upload New Book")
    uploaded_ref = st.sidebar.file_uploader("Select PDF", type=["pdf"], key="ref_uploader")
    if uploaded_ref:
        ref_dir = os.path.join(os.getcwd(), "reference_books")
        os.makedirs(ref_dir, exist_ok=True)
        save_path = os.path.join(ref_dir, uploaded_ref.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_ref.getbuffer())
        st.sidebar.success(f"Saved: {uploaded_ref.name}")
        st.sidebar.info("Click 'Update/Create Cache' to apply.")

    ref_dir = os.path.join(os.getcwd(), "reference_books")
    os.makedirs(ref_dir, exist_ok=True)
    loaded_refs = vlm_engine.load_reference_books(ref_dir)
    if loaded_refs:
        st.sidebar.success(f"📚 Loaded {len(loaded_refs)} Reference Books")
        with st.sidebar.expander("View Loaded Books"):
            for book in loaded_refs:
                st.write(f"- {book}")

# --- Main Page UI ---

st.title("📝 Math Problem Extractor (Hybrid Mode)")
st.markdown("Powered by **Gemini 2.5 Lite** for extraction and **Gemini 2.5 Pro** for solving.")

uploaded_file = st.file_uploader("Upload File", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        num_pages = len(doc)
        process_range = st.radio("Processing Range", ["Current Page", "All Pages"], horizontal=True)
        page_num = st.number_input("Select Page", min_value=1, max_value=num_pages, value=1) if process_range == "Current Page" else 1
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap()
        image = Image.open(io.BytesIO(pix.tobytes("png")))
    else:
        image = Image.open(uploaded_file)
        process_range = "Current Page"

    st.image(image, caption="Uploaded Image", use_container_width=True)
    tab_extract, tab_library = st.tabs(["🔍 Extraction & Solving", "📚 Problem Library"])

    with tab_extract:
        if st.button("Start Extraction"):
            if not vlm_engine:
                st.error("Gemini API Key is required.")
            else:
                with st.spinner("Step 1: Extracting..."):
                    raw_result = vlm_engine.extract_math_problems(image)
                    data = parse_problems_xml(raw_result)
                    
                    if not data["problems"]:
                        st.warning("No problems detected.")
                    else:
                        st.success(f"Detected {len(data['problems'])} problems. Solving in parallel...")
                        status_ui = [st.empty() for _ in data["problems"]]
                        for i, p in enumerate(data["problems"]):
                            status_ui[i].info(f"Problem {p['number']}: 🔄 Processing...")

                        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                            ctx = get_script_run_ctx()
                            futures = {executor.submit(run_with_ctx, ctx, solve_task, p, image, vlm_engine): i for i, p in enumerate(data["problems"])}
                            for future in concurrent.futures.as_completed(futures):
                                i = futures[future]
                                try:
                                    res = future.result()
                                    data["problems"][i].update(res)
                                    status_ui[i].success(f"Problem {data['problems'][i]['number']}: ✅ Done")
                                except Exception as e:
                                    status_ui[i].error(f"Problem {data['problems'][i]['number']}: ❌ Failed ({e})")

                        st.divider()
                        display_results(data, image, vlm_engine)
                        st.session_state.last_results = data
                        
        if "last_results" in st.session_state:
            if st.button("💾 Save All to Library"):
                count = save_to_library(st.session_state.last_results, uploaded_file.name)
                st.success(f"Successfully saved {count} problems!")

    with tab_library:
        library_browser_tab(vlm_engine)

    if uploaded_file.type == "application/pdf" and process_range == "All Pages":
        if st.button("Process All Pages"):
            if not vlm_engine:
                st.error("Gemini API Key is required.")
                st.stop()
            
            all_solve_tasks = []
            
            # --- Stage 1: Global Extraction ---
            extraction_status = st.empty()
            extraction_progress = st.progress(0)
            
            for i in range(num_pages):
                extraction_status.info(f"🔍 Extracting problems from Page {i+1}/{num_pages}...")
                temp_page = doc.load_page(i)
                temp_pix = temp_page.get_pixmap()
                temp_img = Image.open(io.BytesIO(temp_pix.tobytes("png")))
                
                raw_result = vlm_engine.extract_math_problems(temp_img)
                data = parse_problems_xml(raw_result)
                
                if data and "problems" in data:
                    for p in data["problems"]:
                        all_solve_tasks.append({
                            "page": i + 1,
                            "img": temp_img,
                            "prob": p
                        })
                extraction_progress.progress((i + 1) / num_pages)
            
            extraction_status.empty()
            extraction_progress.empty()
            
            if not all_solve_tasks:
                st.warning("No problems detected in the entire document.")
            else:
                st.success(f"Extracted {len(all_solve_tasks)} problems. Starting global solver (4 workers)...")
                
                # --- Stage 2: Individual Status UI ---
                st.markdown("### 🔄 Solving Progress")
                status_placeholders = []
                for t in all_solve_tasks:
                    status_placeholders.append(st.empty())
                
                for idx, t in enumerate(all_solve_tasks):
                    status_placeholders[idx].caption(f"Page {t['page']} - Prob {t['prob']['number']}: ⏳ Waiting...")

                # --- Stage 3: Global Parallel Solving (Safe UI Pattern) ---
                st.markdown("### 🛠 Solving in Progress...")
                solve_progress_bar = st.progress(0)
                solve_status_text = st.empty()
                
                # Shared state for UI updates from main thread
                solve_states = ["⏳ Waiting..."] * len(all_solve_tasks)
                status_lock = threading.Lock()
                
                def solve_worker(idx, task_item):
                    try:
                        with status_lock:
                            solve_states[idx] = "🔄 Solving..."
                        print(f"[THREAD] Starting solve: Prob {task_item['prob']['number']}")
                        
                        res = solve_task(task_item['prob'], task_item['img'], vlm_engine)
                        task_item['prob'].update(res)
                        
                        with status_lock:
                            solve_states[idx] = "✅ Done"
                        print(f"[THREAD] Completed solve: Prob {task_item['prob']['number']}")
                    except Exception as e:
                        with status_lock:
                            solve_states[idx] = f"❌ Error: {str(e)}"
                        print(f"[THREAD ERROR] Prob {task_item['prob'].get('number')}: {e}")
                    return idx

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(solve_worker, i, task) for i, task in enumerate(all_solve_tasks)]
                    
                    total_tasks = len(all_solve_tasks)
                    while True:
                        completed_count = 0
                        # Update all placeholders from the main thread (Safe)
                        with status_lock:
                            for idx, state in enumerate(solve_states):
                                status_placeholders[idx].markdown(f"**Page {all_solve_tasks[idx]['page']} - Prob {all_solve_tasks[idx]['prob']['number']}**: {state}")
                                if "✅ Done" in state or "❌ Error" in state:
                                    completed_count += 1
                        
                        solve_progress_bar.progress(completed_count / total_tasks)
                        solve_status_text.caption(f"Progress: {completed_count}/{total_tasks} problems finished")
                        
                        if completed_count == total_tasks:
                            break
                        
                        time.sleep(0.5) # Poll every 0.5s
                        
                        # Check if all futures are actually done (safety fallback)
                        if all(f.done() for f in futures):
                            break

                st.success(f"🎉 All {total_tasks} problems solved!")
                st.divider()
                
                # Update session state for global saving
                st.session_state.global_results = {"problems": [t['prob'] for t in all_solve_tasks]}
                
                # --- Stage 4: Unified Results Display ---
                for p_num in range(1, num_pages + 1):
                    page_tasks = [t for t in all_solve_tasks if t['page'] == p_num]
                    if page_tasks:
                        st.markdown(f"## Page {p_num}")
                        page_data = {"problems": [t['prob'] for t in page_tasks]}
                        # Use first problem's image for header
                        display_results(page_data, page_tasks[0]['img'], vlm_engine)
                        st.divider()

                if "global_results" in st.session_state:
                    if st.button("💾 Save All Results to Library", key="global_save_btn"):
                        count = save_to_library(st.session_state.global_results, uploaded_file.name)
                        st.success(f"Successfully saved {count} problems across {num_pages} pages!")

