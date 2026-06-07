# -*- coding: utf-8 -*-
"""
模块: 数据库管理
功能: 使用 SQLite 存储车辆检测记录，支持批量写入和查询
"""

import sqlite3
import time
import threading
from .config import CONFIG


class VehicleDB:
    """车辆检测记录数据库"""

    def __init__(self, db_path=None):
        self.db_path = db_path or CONFIG.get('db_path', 'vehicle.db')
        self._cache = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self._batch_size = CONFIG.get('db_batch_size', 50)
        self._flush_interval = CONFIG.get('db_flush_interval', 5.0)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        # 检查旧表结构并迁移
        cursor = conn.execute("PRAGMA table_info(vehicle_record)")
        columns = [row[1] for row in cursor.fetchall()]
        if columns and 'task_id' not in columns:
            conn.execute("DROP TABLE IF EXISTS vehicle_record")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_record (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                track_id INTEGER,
                car_type TEXT,
                avg_speed REAL,
                max_speed REAL,
                lane_num INTEGER,
                min_distance REAL,
                has_smoke INTEGER,
                smoke_grade INTEGER,
                is_overspeed INTEGER,
                frame_count INTEGER,
                create_time TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_record (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE,
                video_name TEXT,
                total_frames INTEGER,
                elapsed_time REAL,
                avg_speed REAL,
                max_speed REAL,
                max_cars INTEGER,
                danger_frames INTEGER,
                smoke_frames INTEGER,
                detection_rate REAL,
                create_time TEXT
            )
        """)
        conn.commit()
        conn.close()

    def add_vehicle(self, task_id, track_id, car_type, avg_speed, max_speed,
                    lane_num, min_distance, has_smoke, smoke_grade,
                    is_overspeed, frame_count):
        record = (
            task_id, track_id, car_type, avg_speed, max_speed,
            lane_num, min_distance, int(has_smoke), smoke_grade,
            int(is_overspeed), frame_count,
            time.strftime("%Y-%m-%d %H:%M:%S")
        )
        with self._lock:
            self._cache.append(record)
        self._try_flush()

    def add_task(self, task_id, video_name, summary):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO task_record
                (task_id, video_name, total_frames, elapsed_time, avg_speed,
                 max_speed, max_cars, danger_frames, smoke_frames,
                 detection_rate, create_time)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                task_id, video_name,
                summary.get('total_frames', 0),
                summary.get('elapsed_time', 0),
                summary.get('avg_speed'),
                summary.get('max_speed'),
                summary.get('max_cars', 0),
                summary.get('danger_frames', 0),
                summary.get('smoke_frames', 0),
                summary.get('detection_rate', 0),
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ))
            conn.commit()
        except Exception as e:
            pass
        finally:
            conn.close()

    def _try_flush(self):
        with self._lock:
            if len(self._cache) < self._batch_size and \
               (time.time() - self._last_flush) < self._flush_interval:
                return
            if not self._cache:
                return
            batch = list(self._cache)
            self._cache.clear()
            self._last_flush = time.time()
        threading.Thread(target=self._do_insert, args=(batch,), daemon=True).start()

    def _do_insert(self, batch):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.executemany("""
                INSERT INTO vehicle_record
                (task_id, track_id, car_type, avg_speed, max_speed,
                 lane_num, min_distance, has_smoke, smoke_grade,
                 is_overspeed, frame_count, create_time)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, batch)
            conn.commit()
            conn.close()
        except Exception as e:
            pass

    def flush_remaining(self):
        with self._lock:
            if not self._cache:
                return
            batch = list(self._cache)
            self._cache.clear()
        self._do_insert(batch)

    def query_tasks(self, limit=50, offset=0):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM task_record ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM task_record").fetchone()[0]
        conn.close()
        return [dict(r) for r in rows], total

    def query_vehicles(self, task_id=None, limit=100, offset=0):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if task_id:
            rows = conn.execute(
                "SELECT * FROM vehicle_record WHERE task_id=? ORDER BY id DESC LIMIT ? OFFSET ?",
                (task_id, limit, offset)
            ).fetchall()
            total = conn.execute(
                "SELECT COUNT(*) FROM vehicle_record WHERE task_id=?", (task_id,)
            ).fetchone()[0]
        else:
            rows = conn.execute(
                "SELECT * FROM vehicle_record ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM vehicle_record").fetchone()[0]
        conn.close()
        return [dict(r) for r in rows], total

    def get_task_stats(self, task_id):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        task = conn.execute(
            "SELECT * FROM task_record WHERE task_id=?", (task_id,)
        ).fetchone()
        vehicles = conn.execute(
            "SELECT * FROM vehicle_record WHERE task_id=? ORDER BY avg_speed DESC",
            (task_id,)
        ).fetchall()
        conn.close()
        if task:
            return dict(task), [dict(v) for v in vehicles]
        return None, []
