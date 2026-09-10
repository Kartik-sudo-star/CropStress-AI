from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import os
from datetime import datetime

app = Flask(__name__, static_folder='.')
CORS(app)

DATA_FILE = "data.json"

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except: pass
    return {
        "messages": [],
        "views": 0,
        "projects": [
            {"id": 1, "title": "Neon Dashboard", "desc": "Real-time analytics with WebSocket, D3.js charts and cyberpunk theme", "icon": "🚀", "likes": 124, "tech": ["React","D3","WebSocket"], "color": "#00d4ff"},
            {"id": 2, "title": "AI Chat Interface", "desc": "Conversational AI with streaming, markdown & voice input", "icon": "🤖", "likes": 89, "tech": ["Next.js","OpenAI","Tailwind"], "color": "#7b2ff7"},
            {"id": 3, "title": "3D Portfolio Engine", "desc": "Three.js + WebGL interactive 3D scenes with particles", "icon": "🎮", "likes": 210, "tech": ["Three.js","GSAP","WebGL"], "color": "#ff2d55"},
            {"id": 4, "title": "Fitness Tracker PWA", "desc": "Offline PWA with push notifications & workout analytics", "icon": "📱", "likes": 67, "tech": ["PWA","IndexedDB","Charts"], "color": "#00d4ff"},
            {"id": 5, "title": "Crypto Tracker", "desc": "Live crypto prices with charts & portfolio manager", "icon": "💹", "likes": 156, "tech": ["Vue","CoinGecko","Chart.js"], "color": "#7b2ff7"},
            {"id": 6, "title": "E-Commerce Neo", "desc": "Next-gen shopping with AR try-on & AI recommendations", "icon": "🛒", "likes": 98, "tech": ["MERN","AR.js","Stripe"], "color": "#ff2d55"},
        ],
        "skills": [
            {"name": "HTML5", "level": 95, "icon": "🌐", "cat": "Frontend"},
            {"name": "CSS3", "level": 90, "icon": "🎨", "cat": "Frontend"},
            {"name": "JavaScript", "level": 85, "icon": "⚡", "cat": "Frontend"},
            {"name": "React", "level": 80, "icon": "⚛️", "cat": "Frontend"},
            {"name": "Python", "level": 88, "icon": "🐍", "cat": "Backend"},
            {"name": "Node.js", "level": 75, "icon": "🗄️", "cat": "Backend"},
            {"name": "Flask", "level": 82, "icon": "🔥", "cat": "Backend"},
            {"name": "MongoDB", "level": 70, "icon": "🍃", "cat": "DB"},
        ],
        "stats": {"projects": 50, "clients": 30, "experience": 5, "satisfaction": 99},
        "timeline": [
            {"id": 1, "time": "2024 — Present", "title": "Senior Frontend Developer", "desc": "Leading UI architecture for enterprise SaaS platforms, mentoring team of 8"},
            {"id": 2, "time": "2022 — 2024", "title": "Frontend Developer", "desc": "Built 20+ responsive web apps, design systems & component libraries"},
            {"id": 3, "time": "2020 — 2022", "title": "Junior Developer", "desc": "Started journey with HTML/CSS/JS, first freelance projects & hackathons"},
        ],
        "hero": {"typed": ["clean & performant web applications","futuristic UI/UX experiences","pixel-perfect responsive designs","interactive animations & 3D scenes"]}
    }

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

data = load_data()
# ensure timeline ids
for idx, t in enumerate(data.get("timeline", [])):
    if "id" not in t:
        t["id"] = idx+1
save_data(data)

# --- ROUTES ---
@app.route('/')
def home():
    data["views"] += 1
    save_data(data)
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

# PROJECTS - FULL CRUD
@app.route('/api/projects', methods=['GET'])
def get_projects():
    return jsonify({"success": True, "projects": data["projects"]})

@app.route('/api/projects/<int:pid>/like', methods=['POST'])
def like_project(pid):
    for p in data["projects"]:
        if p["id"] == pid:
            p["likes"] += 1
            save_data(data)
            return jsonify({"success": True, "likes": p["likes"], "project": p})
    return jsonify({"success": False, "error": "Project not found"}), 404

@app.route('/api/projects', methods=['POST'])
def add_project():
    body = request.get_json() or {}
    new_id = max([p["id"] for p in data["projects"]], default=0) + 1
    proj = {
        "id": new_id,
        "title": body.get("title","Untitled").strip() or "Untitled",
        "desc": body.get("desc","No description"),
        "icon": body.get("icon","✨"),
        "likes": int(body.get("likes",0)),
        "tech": body.get("tech",[]),
        "color": body.get("color","#00d4ff")
    }
    # allow tech as comma string
    if isinstance(proj["tech"], str):
        proj["tech"] = [t.strip() for t in proj["tech"].split(",") if t.strip()]
    data["projects"].append(proj)
    save_data(data)
    return jsonify({"success": True, "project": proj})

@app.route('/api/projects/<int:pid>', methods=['PUT'])
def update_project(pid):
    body = request.get_json() or {}
    for p in data["projects"]:
        if p["id"] == pid:
            if "title" in body: p["title"]=body["title"]
            if "desc" in body: p["desc"]=body["desc"]
            if "icon" in body: p["icon"]=body["icon"]
            if "tech" in body:
                tech = body["tech"]
                if isinstance(tech, str):
                    tech = [t.strip() for t in tech.split(",") if t.strip()]
                p["tech"]=tech
            if "color" in body: p["color"]=body["color"]
            if "likes" in body: p["likes"]=int(body["likes"])
            save_data(data)
            return jsonify({"success": True, "project": p})
    return jsonify({"success": False, "error": "Project not found"}), 404

@app.route('/api/projects/<int:pid>', methods=['DELETE'])
def delete_project(pid):
    global data
    before = len(data["projects"])
    data["projects"] = [p for p in data["projects"] if p["id"] != pid]
    if len(data["projects"]) == before:
        return jsonify({"success": False, "error": "Project not found"}), 404
    save_data(data)
    return jsonify({"success": True})

# SKILLS - FULL CRUD
@app.route('/api/skills', methods=['GET'])
def get_skills():
    return jsonify({"success": True, "skills": data["skills"]})

@app.route('/api/skills', methods=['POST'])
def add_or_update_skill():
    body = request.get_json() or {}
    name = body.get("name","").strip()
    if not name:
        return jsonify({"success": False, "error": "Name required"}), 400
    # if exists update
    for s in data["skills"]:
        if s["name"].lower() == name.lower():
            if "level" in body: s["level"]= max(0,min(100,int(body["level"])))
            if "icon" in body: s["icon"]=body["icon"]
            if "cat" in body: s["cat"]=body["cat"]
            save_data(data)
            return jsonify({"success": True, "skills": data["skills"], "skill": s, "action":"updated"})
    # else create
    new_skill = {
        "name": name,
        "level": max(0,min(100,int(body.get("level",70)))),
        "icon": body.get("icon","✨"),
        "cat": body.get("cat","Other")
    }
    data["skills"].append(new_skill)
    save_data(data)
    return jsonify({"success": True, "skills": data["skills"], "skill": new_skill, "action":"created"})

@app.route('/api/skills/<path:name>', methods=['PUT'])
def update_skill_by_name(name):
    body = request.get_json() or {}
    for s in data["skills"]:
        if s["name"].lower() == name.lower():
            if "level" in body: s["level"]=max(0,min(100,int(body["level"])))
            if "icon" in body: s["icon"]=body["icon"]
            if "cat" in body: s["cat"]=body["cat"]
            if "name" in body and body["name"].strip(): s["name"]=body["name"].strip()
            save_data(data)
            return jsonify({"success": True, "skill": s, "skills": data["skills"]})
    return jsonify({"success": False, "error": "Skill not found"}), 404

@app.route('/api/skills/<path:name>', methods=['DELETE'])
def delete_skill(name):
    before = len(data["skills"])
    data["skills"] = [s for s in data["skills"] if s["name"].lower() != name.lower()]
    if len(data["skills"]) == before:
        return jsonify({"success": False, "error": "Skill not found"}), 404
    save_data(data)
    return jsonify({"success": True, "skills": data["skills"]})

# STATS
@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({"success": True, "stats": data["stats"], "views": data["views"]})

@app.route('/api/stats', methods=['PUT'])
def update_stats():
    body = request.get_json() or {}
    for k in ["projects","clients","experience","satisfaction"]:
        if k in body:
            try: data["stats"][k]=int(body[k])
            except: pass
    save_data(data)
    return jsonify({"success": True, "stats": data["stats"]})

# TIMELINE - FULL CRUD
@app.route('/api/timeline', methods=['GET'])
def get_timeline():
    return jsonify({"success": True, "timeline": data["timeline"]})

@app.route('/api/timeline', methods=['POST'])
def add_timeline():
    body = request.get_json() or {}
    new_id = max([t.get("id",0) for t in data["timeline"]], default=0)+1
    item = {"id": new_id, "time": body.get("time",""), "title": body.get("title",""), "desc": body.get("desc","")}
    if not item["title"]:
        return jsonify({"success": False, "error": "Title required"}), 400
    data["timeline"].insert(0, item)
    save_data(data)
    return jsonify({"success": True, "item": item, "timeline": data["timeline"]})

@app.route('/api/timeline/<int:tid>', methods=['PUT'])
def update_timeline(tid):
    body = request.get_json() or {}
    for t in data["timeline"]:
        if t.get("id")==tid:
            if "time" in body: t["time"]=body["time"]
            if "title" in body: t["title"]=body["title"]
            if "desc" in body: t["desc"]=body["desc"]
            save_data(data)
            return jsonify({"success": True, "item": t})
    return jsonify({"success": False, "error": "Not found"}), 404

@app.route('/api/timeline/<int:tid>', methods=['DELETE'])
def delete_timeline(tid):
    before=len(data["timeline"])
    data["timeline"]=[t for t in data["timeline"] if t.get("id")!=tid]
    if len(data["timeline"])==before:
        return jsonify({"success": False, "error": "Not found"}), 404
    save_data(data)
    return jsonify({"success": True})

# HERO
@app.route('/api/hero', methods=['GET'])
def get_hero():
    return jsonify({"success": True, "hero": data["hero"]})

@app.route('/api/hero', methods=['PUT'])
def update_hero():
    body=request.get_json() or {}
    if "typed" in body:
        typed = body["typed"]
        if isinstance(typed,str):
            typed=[t.strip() for t in typed.split(",") if t.strip()]
        data["hero"]["typed"]=typed
        save_data(data)
    return jsonify({"success": True, "hero": data["hero"]})

# CONTACT & MESSAGES
@app.route('/api/contact', methods=['POST'])
def contact():
    body = request.get_json() or {}
    name = body.get('name','').strip()
    email = body.get('email','').strip()
    message = body.get('message','').strip()
    if not name or not email or not message:
        return jsonify({"success": False, "error": "All fields required"}), 400
    if "@" not in email:
        return jsonify({"success": False, "error": "Invalid email"}), 400
    entry = {"id": max([m.get("id",0) for m in data["messages"]], default=0)+1, "name": name, "email": email, "message": message, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    data["messages"].append(entry)
    save_data(data)
    print(f"[CONTACT] {name} <{email}>")
    return jsonify({"success": True, "message": "Message sent! I'll reply soon", "id": entry["id"]})

@app.route('/api/messages', methods=['GET'])
def get_messages():
    return jsonify({"success": True, "count": len(data["messages"]), "messages": data["messages"][-50:][::-1]})

@app.route('/api/messages/<int:mid>', methods=['DELETE'])
def delete_message(mid):
    before=len(data["messages"])
    data["messages"]=[m for m in data["messages"] if m.get("id")!=mid]
    if len(data["messages"])==before:
        return jsonify({"success": False, "error": "Not found"}), 404
    save_data(data)
    return jsonify({"success": True})

@app.route('/api/messages', methods=['DELETE'])
def clear_messages():
    data["messages"]=[]
    save_data(data)
    return jsonify({"success": True})

@app.route('/api/views', methods=['POST'])
def add_view():
    data["views"] += 1
    save_data(data)
    return jsonify({"success": True, "views": data["views"]})

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "views": data["views"], "projects": len(data["projects"]), "skills": len(data["skills"]), "messages": len(data["messages"]), "time": datetime.now().isoformat()})

@app.route('/api/reset', methods=['POST'])
def reset():
    global data
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    data = load_data()
    return jsonify({"success": True})

if __name__ == '__main__':
    print("="*50)
    print("Futuristic Portfolio Backend v3 - FULL CRUD")
    print("-> http://127.0.0.1:5500")
    print("-> API: http://127.0.0.1:5500/api/health")
    print("-> Projects: GET/POST/PUT/DELETE /api/projects")
    print("-> Skills: GET/POST/PUT/DELETE /api/skills")
    print("-> Stats: GET/PUT /api/stats | Timeline: FULL CRUD | Hero: PUT")
    print("="*50)
    app.run(host='0.0.0.0', port=5500, debug=True)
