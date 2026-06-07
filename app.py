# -*- coding: utf-8 -*-
"""
智能车辆检测分析系统 - Flask Web 应用
实时 MJPEG 流 + 统计 API + 数据查看页面
"""

import os
import time
import atexit
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from src.config import CONFIG
from src.live_processor import LiveProcessor
from src.database import VehicleDB
from src.traffic_db import TrafficDB

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = CONFIG['max_upload_mb'] * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

STREAM_FPS = 25

processor = LiveProcessor()
db = VehicleDB('vehicle.db')
traffic_db = TrafficDB('traffic.db')

# 注入数据库实例到处理器
processor.set_databases(db, traffic_db)

atexit.register(db.flush_remaining)
atexit.register(traffic_db.flush_remaining)
atexit.register(processor._save_task_record)


# ==================== 页面路由 ====================

@app.route('/')
def index():
    return render_template('detect.html', config=CONFIG)


# ==================== 视频上传 ====================

@app.route('/upload', methods=['POST'])
def upload_video():
    f = request.files.get('video')
    mode = request.form.get('mode', 'speed')
    if mode not in ('speed', 'distance', 'smoke'):
        mode = 'speed'
    if f and f.filename:
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(path)
        processor.open_video(path)
        processor.set_mode(mode)
    return redirect('/')


# ==================== MJPEG 视频流 ====================

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = processor.get_frame()
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1.0 / STREAM_FPS)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ==================== API ====================

@app.route('/api/stats')
def api_stats():
    return jsonify(processor.get_stats())


@app.route('/api/switch_mode', methods=['POST'])
def api_switch_mode():
    mode = request.json.get('mode', 'speed')
    if mode in ('speed', 'distance', 'smoke'):
        processor.set_mode(mode)
        return jsonify({'ok': True})
    return jsonify({'ok': False}), 400


# ==================== 数据查看页面 ====================

HTML_PAGE = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="2">
<title>{title}</title>
<style>
body{{background:#0d1117;color:#e6edf3;font-family:'Microsoft YaHei',sans-serif;margin:0;padding:20px}}
h2{{color:#58a6ff;border-left:4px solid #58a6ff;padding-left:12px}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}
th,td{{border:1px solid #30363d;padding:10px 14px;text-align:left}}
th{{background:#161b22;color:#58a6ff}} td{{background:#0d1117}}
.v{{font-size:28px;font-weight:bold;color:#58a6ff}}
.sub{{color:#8b949e;font-size:13px;margin-top:4px}}
.tag{{display:inline-block;background:#161b22;border:1px solid #30363d;border-radius:6px;padding:3px 10px;margin:2px;font-size:13px}}
.ok{{color:#4ade80}} .warn{{color:#f87171}}
a{{color:#58a6ff;text-decoration:none}}a:hover{{text-decoration:underline}}
</style></head><body>
<a href="/">← 返回主界面</a> &nbsp;|&nbsp; <a href="/realtime_speed" target="_blank">车速</a> &nbsp;
<a href="/flow_data" target="_blank">车流</a> &nbsp; <a href="/car_type" target="_blank">车型</a> &nbsp;
<a href="/history_speed" target="_blank">历史</a> &nbsp; <a href="/report" target="_blank">报告</a>
<h2>{title}</h2>{body}
<p class="sub">数据每 2 秒自动刷新 | 智能车辆检测系统</p></body></html>"""


@app.route('/realtime_speed')
def page_realtime_speed():
    stats = processor.get_stats()
    # 从当前跟踪的车辆获取实时速度
    speeds = []
    if processor.speed_calc:
        for vid, spd in processor.speed_calc.smoothed.items():
            if spd > 0:
                speeds.append(round(spd, 1))
    speeds.sort(reverse=True)
    rows = ""
    for i, spd in enumerate(speeds[:30]):
        cls = 'ok' if spd < 120 else 'warn'
        rows += f"<tr><td>#{i + 1}</td><td class='{cls}'>{spd:.1f} km/h</td></tr>"
    if not rows:
        rows = "<tr><td colspan='2'>暂无测速数据，请等待车辆进入画面</td></tr>"
    body = f"""<p><span class="v">{stats['avg_speed']}</span> km/h <span class="sub">当前平均车速</span></p>
    <p class="sub">当前跟踪: {len(speeds)} 辆 | 检测置信度: {CONFIG['detect_conf']} | 限速: {CONFIG['speed_limit']} km/h</p>
    <table><tr><th>序号</th><th>车速</th></tr>{rows}</table>"""
    return HTML_PAGE.format(title="实时车速监测", body=body)


@app.route('/flow_data')
def page_flow_data():
    import json as _json
    stats = processor.get_stats()
    names = stats.get('lane_names', ['车道1'])
    counts = stats.get('lane_counts', [0])
    pcts = stats.get('lane_pcts', [0])
    total = stats.get('lane_total', 0)
    lh_json = _json.dumps(stats.get('lane_history', []))
    names_json = _json.dumps(names)
    colors = ['#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#ec4899', '#06b6d4', '#f97316', '#84cc16']
    bar_html = ''
    table_rows = ''
    for i, (n, c, p) in enumerate(zip(names, counts, pcts)):
        co = colors[i % len(colors)]
        bar_html += f'<div style="width:{max(p,2)}%;background:{co}">{c}辆</div>'
        table_rows += f'<tr><td style="color:{co}">{n}</td><td>{c} 辆</td><td>{p}%</td></tr>'
    table_rows += f'<tr><td style="color:#c084fc">合计</td><td>{total} 辆</td><td>100%</td></tr>'
    return ('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta http-equiv="refresh" content="2"><title>车流统计</title>'
            '<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>'
            '<style>'
            'body{background:#0d1117;color:#e6edf3;font-family:"Microsoft YaHei",sans-serif;margin:0;padding:20px}'
            'h2{color:#58a6ff;border-left:4px solid #58a6ff;padding-left:12px}'
            'table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #30363d;padding:10px 14px;text-align:left}'
            'th{background:#161b22;color:#58a6ff}td{background:#0d1117}.v{font-size:28px;font-weight:bold;color:#58a6ff}'
            '.sub{color:#8b949e;font-size:13px;margin-top:4px}.warn{color:#f87171}'
            'a{color:#58a6ff;text-decoration:none}a:hover{text-decoration:underline}'
            '.bar{display:flex;height:40px;border-radius:8px;overflow:hidden;margin:12px 0}'
            '.bar div{display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:#0d1117}'
            '#laneChart{width:100%;height:280px}'
            '</style></head><body>'
            '<a href="/">← 返回主界面</a>'
            '<h2>车流统计 - 车道流量分析</h2>'
            f'<p><span class="v">{stats["total_vehicles"]}</span> 辆 <span class="sub">累计通行车辆</span></p>'
            f'<div class="bar">{bar_html}</div>'
            '<table><tr><th>车道</th><th>当前车辆</th><th>占比</th></tr>'
            f'{table_rows}</table>'
            '<h3 style="color:#8ecae6;font-size:14px;margin:16px 0 8px">车流量趋势</h3>'
            '<div id="laneChart"></div>'
            '<script>'
            'var c=echarts.init(document.getElementById("laneChart"));'
            f'var lh={lh_json};var lNames={names_json};'
            'var cats=lh.map(function(x){return "-"+x.t+"s"});'
            'var lColors=["#3b82f6","#22c55e","#f59e0b","#a855f7","#ec4899","#06b6d4","#f97316","#84cc16"];'
            'var series=[];'
            'for(var i=0;i<lNames.length;i++){series.push({name:lNames[i],type:"bar",stack:"lane",'
            'data:lh.map(function(x){return x["l"+i]||0}),itemStyle:{color:lColors[i%lColors.length]},barWidth:"25%"});}'
            'series.push({name:"总流量",type:"line",yAxisIndex:1,'
            'data:lh.map(function(x){return x.total||0}),smooth:true,lineStyle:{color:"#c084fc",width:2},'
            'areaStyle:{color:{type:"linear",x:0,y:0,x2:0,y2:1,colorStops:[{offset:0,color:"rgba(192,132,252,0.2)"},{offset:1,color:"rgba(192,132,252,0)"}]}},'
            'itemStyle:{color:"#c084fc"}});'
            'c.setOption({tooltip:{trigger:"axis"},legend:{data:series.map(function(s){return s.name}),textStyle:{color:"#8b949e"},top:0},'
            'grid:{left:50,right:50,top:30,bottom:30},'
            'xAxis:{type:"category",data:cats,axisLabel:{color:"#8b949e",fontSize:10}},'
            'yAxis:[{type:"value",name:"车道",axisLabel:{color:"#8b949e"},splitLine:{lineStyle:{color:"#21262d"}}},'
            '{type:"value",name:"总量",axisLabel:{color:"#8b949e"},splitLine:{show:false}}],'
            'series:series});'
            '</script></body></html>')


@app.route('/car_type')
def page_car_type():
    stats = processor.get_stats()
    body = f"""<p><span class="v">{stats['total_vehicles']}</span> 辆 <span class="sub">累计检测车辆</span></p>
    <p>检测类别 (COCO): <span class="tag">car(2)</span><span class="tag">motorcycle(3)</span><span class="tag">bus(5)</span><span class="tag">truck(7)</span></p>
    <table><tr><th>车型</th><th>COCO ID</th><th>说明</th></tr>
    <tr><td>轿车 (car)</td><td>2</td><td>小型乘用车</td></tr>
    <tr><td>摩托车 (motorcycle)</td><td>3</td><td>摩托车</td></tr>
    <tr><td>大巴 (bus)</td><td>5</td><td>公共汽车/长途客车</td></tr>
    <tr><td>货车 (truck)</td><td>7</td><td>卡车/货车</td></tr></table>
    <p class="sub">基于 YOLOv8s COCO 预训练模型自动分类</p>"""
    return HTML_PAGE.format(title="车型统计", body=body)


@app.route('/history_speed')
def page_history_speed():
    stats = processor.get_stats()
    speeds = []
    if processor.speed_calc:
        for spd in processor.speed_calc.smoothed.values():
            if spd > 0:
                speeds.append(round(spd, 1))
    if speeds:
        avg = round(sum(speeds) / len(speeds), 1)
        mn, mx = min(speeds), max(speeds)
        seg = {"0-30": 0, "30-60": 0, "60-90": 0, "90+": 0}
        for s in speeds:
            if s < 30:
                seg["0-30"] += 1
            elif s < 60:
                seg["30-60"] += 1
            elif s < 90:
                seg["60-90"] += 1
            else:
                seg["90+"] += 1
        seg_rows = "".join(f"<tr><td>{k} km/h</td><td>{v} 辆</td></tr>" for k, v in seg.items())
    else:
        avg = mn = mx = 0
        seg_rows = "<tr><td colspan='2'>暂无数据</td></tr>"
    body = f"""<p><span class="v">{avg}</span> km/h <span class="sub">当前平均车速</span></p>
    <table><tr><th>指标</th><th>数值</th></tr>
    <tr><td>速度范围</td><td>{mn} ~ {mx} km/h</td></tr>
    <tr><td>当前跟踪车辆</td><td>{len(speeds)}</td></tr>
    <tr><td>累计通行车辆</td><td>{stats['total_vehicles']} 辆</td></tr>
    <tr><td>累计危险车辆</td><td class="warn">{stats['danger_vehicles']} 辆</td></tr>
    <tr><td>平均车距</td><td>{stats['avg_distance']} 米</td></tr>
    <tr><td>黑烟超标</td><td class="warn">{stats['smoke_exceed']} 辆次</td></tr></table>
    <h3 style="color:#58a6ff;font-size:15px">分段车速统计</h3>
    <table><tr><th>速度区间</th><th>车辆数</th></tr>{seg_rows}</table>"""
    return HTML_PAGE.format(title="历史车速统计", body=body)


@app.route('/db_records')
def page_db_records():
    page = request.args.get('page', 1, type=int)
    per_page = 50
    offset = (page - 1) * per_page
    records, total = db.query_vehicles(limit=per_page, offset=offset)
    total_pages = max(1, (total + per_page - 1) // per_page)
    rows = ""
    for r in records:
        rows += (f"<tr><td>{r['id']}</td><td>#{r.get('track_id','')}</td><td>{r.get('car_type','')}</td>"
                 f"<td>{r.get('avg_speed',0)} km/h</td><td>{r.get('lane_num',0)}</td>"
                 f"<td>{r.get('min_distance',0)} m</td><td>{'是' if r.get('has_smoke') else '否'}</td>"
                 f"<td>{r.get('create_time','')}</td></tr>")
    if not rows:
        rows = "<tr><td colspan='8'>暂无入库记录</td></tr>"
    body = f"""<p><span class="v">{total}</span> 条 <span class="sub">车辆检测记录</span></p>
    <table><tr><th>ID</th><th>车辆编号</th><th>车型</th><th>平均车速</th><th>车道</th><th>最小车距</th><th>黑烟</th><th>入库时间</th></tr>
    {rows}</table>
    <div style="margin-top:16px">
    {"<a href='/db_records?page=" + str(page - 1) + "' class='tag'>← 上一页</a>" if page > 1 else ""}
    <span class="tag">第 {page} / {total_pages} 页</span>
    {"<a href='/db_records?page=" + str(page + 1) + "' class='tag'>下一页 →</a>" if page < total_pages else ""}
    </div>"""
    return HTML_PAGE.format(title="历史入库记录", body=body)


@app.route('/traffic_db')
def page_traffic_db():
    page = request.args.get('page', 1, type=int)
    tab = request.args.get('tab', 'vehicles')
    per_page = 30
    offset = (page - 1) * per_page
    summary = traffic_db.get_summary()
    if tab == 'violations':
        rows, total = traffic_db.query_violations(limit=per_page, offset=offset)
        headers = ['ID', '车辆编号', '违章类型', '发生时间', '车道']
        cols = ['id', 'track_id', 'violation_type', 'happen_time', 'lane_id']
        title = '违章记录'
    elif tab == 'lanes':
        rows, total = traffic_db.query_lane_stats(limit=per_page, offset=offset)
        headers = ['ID', '统计时间', '车道', '轿车', '公交', '平均车速']
        cols = ['id', 'stat_min', 'lane_id', 'car_num', 'bus_num', 'avg_lane_speed']
        title = '车道统计'
    else:
        rows, total = traffic_db.query_vehicles(limit=per_page, offset=offset)
        headers = ['ID', '车辆编号', '车型', '平均车速', '入场时间', '离场时间', '车道']
        cols = ['id', 'track_id', 'car_type', 'avg_speed', 'enter_time', 'leave_time', 'lane_id']
        title = '车辆信息'
        tab = 'vehicles'
    total_pages = max(1, (total + per_page - 1) // per_page)
    trows = ""
    for r in rows:
        trows += "<tr>" + "".join(f"<td>{r.get(c,'')}</td>" for c in cols) + "</tr>"
    if not trows:
        trows = f"<tr><td colspan='{len(headers)}'>暂无数据</td></tr>"
    tabs = [('vehicles', '车辆信息'), ('violations', '违章记录'), ('lanes', '车道统计')]
    tab_html = "".join(
        f"<a href='/traffic_db?tab={t[0]}' class='tag' style='{'background:#1a73e8;color:#fff' if t[0]==tab else ''}'>{t[1]}</a> "
        for t in tabs
    )
    body = f"""<p><span class="v">{summary['total_vehicles']}</span> 辆车 &nbsp;
    <span class="v">{summary['total_violations']}</span> 违章 &nbsp;
    <span class="v">{summary['total_lane_records']}</span> 车道记录 &nbsp;
    平均 <span class="v">{summary['global_avg_speed']}</span> km/h</p>
    <div style="margin:12px 0">{tab_html}</div>
    <h3>{title}（共 {total} 条）</h3>
    <table><tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr>{trows}</table>
    <div style="margin-top:16px">
    {"<a href='/traffic_db?tab=" + tab + "&page=" + str(page-1) + "' class='tag'>← 上一页</a>" if page > 1 else ""}
    <span class="tag">第 {page} / {total_pages} 页</span>
    {"<a href='/traffic_db?tab=" + tab + "&page=" + str(page+1) + "' class='tag'>下一页 →</a>" if page < total_pages else ""}
    </div>"""
    return HTML_PAGE.format(title="交通数据库 - " + title, body=body)


@app.route('/report')
def page_report():
    stats = processor.get_stats()
    body = f"""<p class="sub">视频: {stats['current_video'] or '--'} | 帧数: {stats['total_frames']} | 模式: {stats['current_mode']}</p>
    <table><tr><th>指标</th><th>数值</th></tr>
    <tr><td>累计通行车辆</td><td><span class="v">{stats['total_vehicles']}</span> 辆</td></tr>
    <tr><td>平均车速</td><td><span class="v">{stats['avg_speed']}</span> km/h</td></tr>
    <tr><td>累计危险车辆</td><td class="warn">{stats['danger_vehicles']} 辆</td></tr>
    <tr><td>平均车距</td><td>{stats['avg_distance']} 米</td></tr>
    <tr><td>黑烟超标总数</td><td class="warn">{stats['smoke_exceed']} 辆次</td></tr>
    <tr><td>平均林格曼等级</td><td>{stats['avg_ringelmann']} 级</td></tr></table>"""
    return HTML_PAGE.format(title="检测报告", body=body)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
