import sqlite3
import json
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
import threading


@dataclass
class PreferenceRecord:
    """User preference record"""
    id: Optional[int] = None
    user_id: str = ""
    session_id: str = ""
    question: str = ""
    question_hash: str = ""
    chosen_style: str = ""
    generation_sequence: List[str] = field(default_factory=list)
    interaction_time: float = 0.0
    context_length: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPreferenceProfile:
    """User preference profile"""
    user_id: str
    style_preferences: Dict[str, float]
    preferred_style: str
    confidence_score: float
    has_conflict: bool
    total_interactions: int
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class PreferenceDatabase:
    """Preference learning database with simple algorithms"""

    def __init__(self, db_path: str = "./data/database/preferences.db", config: Dict[str, Any] = None):
        self.db_path = Path(db_path)
        self.config = config or {}
        self.logger = logging.getLogger("preference_db")
        self._db_lock = threading.RLock()

        # Simple learning parameters
        self.min_samples = self.config.get("preference_learning.min_samples", 3)
        self.conflict_threshold = self.config.get("preference_learning.conflict_threshold", 0.3)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

        self.logger.info(f"Preference database initialized: {self.db_path}")

    def _init_database(self):
        """Initialize database tables"""
        with self._db_lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            try:
                # Preference records table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS preference_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        question TEXT NOT NULL,
                        question_hash TEXT NOT NULL,
                        chosen_style TEXT NOT NULL,
                        generation_sequence TEXT NOT NULL,
                        interaction_time REAL DEFAULT 0.0,
                        context_length INTEGER DEFAULT 0,
                        timestamp TEXT NOT NULL,
                        metadata TEXT DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON preference_records(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON preference_records(timestamp)')

                # User profiles table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id TEXT PRIMARY KEY,
                        style_preferences TEXT NOT NULL,
                        preferred_style TEXT NOT NULL,
                        confidence_score REAL DEFAULT 0.0,
                        has_conflict BOOLEAN DEFAULT FALSE,
                        total_interactions INTEGER DEFAULT 0,
                        last_updated TEXT NOT NULL,
                        metadata TEXT DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # System stats table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        stat_name TEXT NOT NULL UNIQUE,
                        stat_value TEXT NOT NULL,
                        last_updated TEXT NOT NULL,
                        metadata TEXT DEFAULT '{}'
                    )
                ''')

                conn.commit()
                self.logger.info("Database tables initialized successfully")

            except Exception as e:
                conn.rollback()
                self.logger.error(f"Failed to initialize database: {e}")
                raise

            finally:
                conn.close()

    def _hash_question(self, question: str) -> str:
        """Generate question hash for similarity detection"""
        normalized = question.lower().strip()
        import re
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()

    def record_preference(self, record: PreferenceRecord) -> int:
        """Record user preference with simple processing"""
        with self._db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            try:
                if not record.question_hash:
                    record.question_hash = self._hash_question(record.question)

                cursor.execute('''
                    INSERT INTO preference_records (
                        user_id, session_id, question, question_hash, chosen_style,
                        generation_sequence, interaction_time, context_length,
                        timestamp, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.user_id,
                    record.session_id,
                    record.question,
                    record.question_hash,
                    record.chosen_style,
                    json.dumps(record.generation_sequence),
                    record.interaction_time,
                    record.context_length,
                    record.timestamp.isoformat(),
                    json.dumps(record.metadata)
                ))

                record_id = cursor.lastrowid
                conn.commit()

                # Update user preferences with simple algorithm
                self._update_user_preferences_simple(record.user_id)

                self.logger.debug(f"Recorded preference for user {record.user_id}: {record.chosen_style}")
                return record_id

            except Exception as e:
                conn.rollback()
                self.logger.error(f"Failed to record preference: {e}")
                raise

            finally:
                conn.close()

    def _update_user_preferences_simple(self, user_id: str):
        """Simple preference learning algorithm - basic counting"""
        with self._db_lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            try:
                # Get user's preference records
                cursor.execute('''
                    SELECT chosen_style, timestamp, interaction_time
                    FROM preference_records 
                    WHERE user_id = ? 
                    ORDER BY timestamp ASC
                ''', (user_id,))

                records = cursor.fetchall()

                if len(records) < self.min_samples:
                    self.logger.debug(f"Insufficient samples for user {user_id}: {len(records)}")
                    return

                # Simple counting algorithm
                style_counts = {"balanced": 0, "detailed_policy": 0, "practical_guide": 0}
                total_interactions = len(records)

                for record in records:
                    style_counts[record['chosen_style']] += 1

                # Calculate preferences as percentages
                style_preferences = {
                    style: count / total_interactions
                    for style, count in style_counts.items()
                }

                # Find preferred style
                preferred_style = max(style_preferences, key=style_preferences.get)
                confidence_score = style_preferences[preferred_style]

                # Simple conflict detection
                sorted_prefs = sorted(style_preferences.values(), reverse=True)
                has_conflict = (len(sorted_prefs) >= 2 and
                                sorted_prefs[0] - sorted_prefs[1] < self.conflict_threshold)

                # Create profile
                profile = UserPreferenceProfile(
                    user_id=user_id,
                    style_preferences=style_preferences,
                    preferred_style=preferred_style,
                    confidence_score=confidence_score,
                    has_conflict=has_conflict,
                    total_interactions=total_interactions,
                    last_updated=datetime.now()
                )

                # Save profile
                cursor.execute('''
                    INSERT OR REPLACE INTO user_profiles (
                        user_id, style_preferences, preferred_style, confidence_score,
                        has_conflict, total_interactions, last_updated, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    profile.user_id,
                    json.dumps(profile.style_preferences),
                    profile.preferred_style,
                    profile.confidence_score,
                    profile.has_conflict,
                    profile.total_interactions,
                    profile.last_updated.isoformat(),
                    json.dumps(profile.metadata)
                ))

                conn.commit()

                self.logger.info(
                    f"Updated preferences for user {user_id}: {preferred_style} (confidence: {confidence_score:.3f})")

            except Exception as e:
                conn.rollback()
                self.logger.error(f"Failed to update user preferences: {e}")
                raise

            finally:
                conn.close()

    def get_user_preferences(self, user_id: str) -> Optional[UserPreferenceProfile]:
        """Get user preference profile"""
        with self._db_lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            try:
                cursor.execute(
                    "SELECT * FROM user_profiles WHERE user_id = ?",
                    (user_id,)
                )

                row = cursor.fetchone()
                if not row:
                    return None

                return UserPreferenceProfile(
                    user_id=row['user_id'],
                    style_preferences=json.loads(row['style_preferences']),
                    preferred_style=row['preferred_style'],
                    confidence_score=row['confidence_score'],
                    has_conflict=bool(row['has_conflict']),
                    total_interactions=row['total_interactions'],
                    last_updated=datetime.fromisoformat(row['last_updated']),
                    metadata=json.loads(row['metadata'])
                )

            except Exception as e:
                self.logger.error(f"Failed to get user preferences: {e}")
                return None

            finally:
                conn.close()

    def get_preferred_style(self, user_id: str) -> str:
        """Get user's preferred style, default to balanced"""
        profile = self.get_user_preferences(user_id)
        return profile.preferred_style if profile else "balanced"

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system-wide statistics"""
        with self._db_lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            try:
                # Overall stats
                cursor.execute('''
                    SELECT 
                        COUNT(DISTINCT user_id) as total_users,
                        COUNT(*) as total_interactions,
                        AVG(interaction_time) as avg_interaction_time
                    FROM preference_records
                ''')

                overall_stats = cursor.fetchone()

                # Style distribution
                cursor.execute('''
                    SELECT chosen_style, COUNT(*) as count
                    FROM preference_records 
                    GROUP BY chosen_style
                ''')

                style_stats = {}
                total = overall_stats['total_interactions'] or 1

                for row in cursor.fetchall():
                    style_stats[row['chosen_style']] = {
                        "count": row['count'],
                        "percentage": (row['count'] / total) * 100
                    }

                # Users with conflicts
                cursor.execute('''
                    SELECT COUNT(*) as users_with_conflicts
                    FROM user_profiles 
                    WHERE has_conflict = TRUE
                ''')

                conflict_count = cursor.fetchone()['users_with_conflicts']

                return {
                    "total_users": overall_stats['total_users'] or 0,
                    "total_interactions": overall_stats['total_interactions'] or 0,
                    "avg_interaction_time": overall_stats['avg_interaction_time'] or 0.0,
                    "users_with_conflicts": conflict_count or 0,
                    "style_distribution": style_stats,
                    "conflict_rate": (conflict_count or 0) / max(overall_stats['total_users'] or 1, 1)
                }

            except Exception as e:
                self.logger.error(f"Failed to get system statistics: {e}")
                return {"error": str(e)}

            finally:
                conn.close()

    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for specific user"""
        with self._db_lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            try:
                # Basic stats
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_interactions,
                        AVG(interaction_time) as avg_interaction_time,
                        MIN(timestamp) as first_interaction,
                        MAX(timestamp) as last_interaction
                    FROM preference_records 
                    WHERE user_id = ?
                ''', (user_id,))

                basic_stats = cursor.fetchone()

                # Style distribution
                cursor.execute('''
                    SELECT chosen_style, COUNT(*) as count
                    FROM preference_records 
                    WHERE user_id = ?
                    GROUP BY chosen_style
                ''', (user_id,))

                style_distribution = dict(cursor.fetchall())

                # Get profile
                profile = self.get_user_preferences(user_id)

                return {
                    "user_id": user_id,
                    "total_interactions": basic_stats['total_interactions'] or 0,
                    "avg_interaction_time": basic_stats['avg_interaction_time'] or 0.0,
                    "first_interaction": basic_stats['first_interaction'],
                    "last_interaction": basic_stats['last_interaction'],
                    "style_distribution": style_distribution,
                    "preference_profile": asdict(profile) if profile else None
                }

            except Exception as e:
                self.logger.error(f"Failed to get user statistics: {e}")
                return {"user_id": user_id, "error": str(e)}

            finally:
                conn.close()

    def reset_user_preferences(self, user_id: str) -> bool:
        """Reset all preferences for a user"""
        with self._db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            try:
                cursor.execute("DELETE FROM preference_records WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))

                conn.commit()
                self.logger.info(f"Reset preferences for user {user_id}")
                return True

            except Exception as e:
                conn.rollback()
                self.logger.error(f"Failed to reset user preferences: {e}")
                return False

            finally:
                conn.close()

    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all data for a user"""
        with self._db_lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            try:
                # Get preference records
                cursor.execute('''
                    SELECT * FROM preference_records 
                    WHERE user_id = ?
                    ORDER BY timestamp
                ''', (user_id,))

                preference_records = []
                for row in cursor.fetchall():
                    record_dict = dict(row)
                    record_dict['generation_sequence'] = json.loads(record_dict['generation_sequence'])
                    record_dict['metadata'] = json.loads(record_dict['metadata'])
                    preference_records.append(record_dict)

                # Get user profile
                profile = self.get_user_preferences(user_id)
                profile_dict = asdict(profile) if profile else None

                # Get user statistics
                user_stats = self.get_user_statistics(user_id)

                return {
                    "user_id": user_id,
                    "export_timestamp": datetime.now().isoformat(),
                    "preference_records": preference_records,
                    "user_profile": profile_dict,
                    "statistics": user_stats
                }

            except Exception as e:
                self.logger.error(f"Failed to export user data: {e}")
                return {"error": str(e)}

            finally:
                conn.close()

    def close(self):
        """Close database connections"""
        self.logger.info("Preference database service shutdown")


def create_preference_database(db_path: str = "./data/database/preferences.db",
                               config: Dict[str, Any] = None) -> PreferenceDatabase:
    """Create preference database instance"""
    return PreferenceDatabase(db_path, config)