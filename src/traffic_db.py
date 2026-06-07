# -*- coding: utf-8 -*-
"""
模块: 交通数据库 (traffic.db)
功能: 三张表 - vehicle_info / lane_min_stat / violation_info
入库规则: 内存缓存 -> 满12条或3秒 -> 子线程批量executemany入库
"""

import sqlite3
import time
import threading
from datetime import datetime


class TrafficDB:
    """交通检测数据库 - 子线程异步批量入库，绝不阻塞主线程"""

    def __init__(self, db_path='traffic.db'):
        self.db_path = db_path
        self._vehicle_cache = []
        self._lane_cache = []
        self._violation_cache = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self._batch_size = 10
        self._flush_interval = 3.0
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                car_type TEXT,
                avg_speed REAL,
                enter_time TEXT,
                leave_time TEXT,
                lane_id INTEGER,
                save_time TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lane_min_stat (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stat_min TEXT,
                lane_id INTEGER,
                car_num INTEGER DEFAULT 0,
                bus_num INTEGER DEFAULT 0,
                avg_lane_speed REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS violation_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                violation_type TEXT,
                happen_time TEXT,
                lane_id INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def add_vehicle(self, track_id, car_type, avg_speed, enter_time, leave_time, lane_id):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = (track_id, car_type, avg_speed, enter_time, leave_time, lane_id, now)
        with self._lock:
            self._vehicle_cache.append(record)
        self._try_flush()

    def add_lane_stat(self, stat_min, lane_id, car_num, bus_num, avg_lane_speed):
        record = (stat_min, lane_id, car_num, bus_num, avg_lane_speed)
        with self._lock:
            self._lane_cache.append(record)
        self._try_flush()

    def add_violation(self, track_id, violation_type, happen_time, lane_id):
        record = (track_id, violation_type, happen_time, lane_id)
        with self._lock:
            self._violation_cache.append(record)
        self._try_flush()

    def _try_flush(self):
        with self._lock:
            total = len(self._vehicle_cache) + len(self._lane_cache) + len(self._violation_cache)
            if total < self._batch_size and (time.time() - self._last_flush) < self._flush_interval:
                return
            if total == 0:
                return
            v_batch = list(self._vehicle_cache)
            l_batch = list(self._lane_cache)
            vi_batch = list(self._violation_cache)
            self._vehicle_cache.clear()
            self._lane_cache.clear()
            self._violation_cache.clear()
            self._last_flush = time.time()
        threading.Thread(target=self._do_insert, args=(v_batch, l_batch, vi_batch), daemon=True).start()

    def _do_insert(self, v_batch, l_batch, vi_batch):
        try:
            conn = sqlite3.connect(self.db_path)
            if v_batch:
                conn.executemany(
                    "INSERT INTO vehicle_info (track_id,car_type,avg_speed,enter_time,leave_time,lane_id,save_time) VALUES (?,?,?,?,?,?,?)",
                    v_batch)
            if l_batch:
                conn.executemany(
                    "INSERT INTO lane_min_stat (stat_min,lane_id,car_num,bus_num,avg_lane_speed) VALUES (?,?,?,?,?)",
                    l_batch)
            if vi_batch:
                conn.executemany(
                    "INSERT INTO violation_info (track_id,violation_type,happen_time,lane_id) VALUES (?,?,?,?)",
                    vi_batch)
            conn.commit()
            conn.close()
        except Exception:
            pass

    def flush_remaining(self):
        with self._lock:
            v_batch = list(self._vehicle_cache)
            l_batch = list(self._lane_cache)
            vi_batch = list(self._violation_cache)
            self._vehicle_cache.clear()
            self._lane_cache.clear()
            self._violation_cache.clear()
        if v_batch or l_batch or vi_batch:
            self._do_insert(v_batch, l_batch, vi_batch)

    # ── 查询接口 (网页历史统计用) ──

    def query_vehicles(self, limit=100, offset=0):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM vehicle_info ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM vehicle_info").fetchone()[0]
        conn.close()
        return [dict(r) for r in rows], total

    def query_lane_stats(self, limit=100, offset=0):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM lane_min_stat ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM lane_min_stat").fetchone()[0]
        conn.close()
        return [dict(r) for r in rows], total

    def query_violations(self, limit=100, offset=0):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM violation_info ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM violation_info").fetchone()[0]
        conn.close()
        return [dict(r) for r in rows], total

    def get_summary(self):
        conn = sqlite3.connect(self.db_path)
        v_count = conn.execute("SELECT COUNT(*) FROM vehicle_info").fetchone()[0]
        vi_count = conn.execute("SELECT COUNT(*) FROM violation_info").fetchone()[0]
        l_count = conn.execute("SELECT COUNT(*) FROM lane_min_stat").fetchone()[0]
        avg_speed = conn.execute("SELECT AVG(avg_speed) FROM vehicle_info WHERE avg_speed > 0").fetchone()[0]
        conn.close()
        return {
            'total_vehicles': v_count,
            'total_violations': vi_count,
            'total_lane_records': l_count,
            'global_avg_speed': round(avg_speed, 1) if avg_speed else 0,
        }
