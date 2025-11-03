import mysql.connector
from datetime import datetime
from contextlib import contextmanager

@contextmanager
def get_db_connection(db_config):
    conn = None
    try:
        conn = mysql.connector.connect(**db_config)
        yield conn
    except mysql.connector.Error as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def _ensure_table(cursor):
    """Create the alert table if it is missing."""

    create_query = """
        CREATE TABLE IF NOT EXISTS kpi_alertes (
            id INT AUTO_INCREMENT PRIMARY KEY,
            Site VARCHAR(50) NOT NULL,
            PDC VARCHAR(50) NOT NULL,
            type_erreur VARCHAR(100) NOT NULL,
            detection DATETIME NOT NULL,
            occurrences_12h INT NOT NULL,
            moment VARCHAR(20) NULL,
            evi_code VARCHAR(50) NULL,
            downstream_code_pc VARCHAR(50) NULL,
            UNIQUE KEY uniq_site_pdc_type (Site, PDC, type_erreur)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """

    cursor.execute(create_query)


def _normalise_alert_row(row):
    """Validate and normalise alert values before insertion."""

    site = str(row.get("Site", "")).strip()[:50]
    pdc = str(row.get("PDC", "")).strip()[:50]
    error_type = str(row.get("Type d'erreur", "")).strip()[:100]

    detection = row.get("Détection")
    if not isinstance(detection, datetime):
        if detection:
            try:
                detection = datetime.fromisoformat(str(detection))
            except Exception as exc:
                raise ValueError("Chaque alerte doit contenir une date de détection valide") from exc
        else:
            detection = None
    if detection is None:
        raise ValueError("Chaque alerte doit contenir une date de détection valide")

    occurrences = int(row.get("Occurrences sur 12h", 0) or 0)

    moment = row.get("Moment")
    moment = str(moment).strip()[:20] if moment else None

    evi_code = row.get("EVI Code")
    evi_code = str(evi_code).strip()[:50] if evi_code else None

    downstream = row.get("Downstream Code PC")
    downstream = str(downstream).strip()[:50] if downstream else None

    return (site, pdc, error_type, detection, occurrences, moment, evi_code, downstream)


def save_alerts_to_db(alert_rows, db_config=None):
    if db_config is None:
        db_config = {
            "host": "162.19.251.55",
            "port": 3306,
            "user": "nidec",
            "password": "MaV38f5xsGQp83",
            "database": "Charges",
        }

    try:
        with get_db_connection(db_config) as conn:
            cursor = conn.cursor()
            try:
                _ensure_table(cursor)

                cursor.execute("DELETE FROM kpi_alertes")

                if not alert_rows:
                    conn.commit()
                    return {"success": True, "rows_affected": 0, "error": None}

                insert_query = """
                    INSERT INTO kpi_alertes
                    (Site, PDC, type_erreur, detection, occurrences_12h, moment, evi_code, downstream_code_pc)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        detection = VALUES(detection),
                        occurrences_12h = VALUES(occurrences_12h),
                        moment = VALUES(moment),
                        evi_code = VALUES(evi_code),
                        downstream_code_pc = VALUES(downstream_code_pc)
                """

                data = [_normalise_alert_row(row) for row in alert_rows]

                cursor.executemany(insert_query, data)
                conn.commit()

                rows_affected = cursor.rowcount

                return {
                    "success": True,
                    "rows_affected": rows_affected,
                    "error": None
                }
            finally:
                cursor.close()

    except mysql.connector.Error as e:
        return {
            "success": False,
            "rows_affected": 0,
            "error": f"Erreur MySQL: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "rows_affected": 0,
            "error": f"Erreur inattendue: {e}"
        }


def get_recent_alerts(days=7, db_config=None):
    if db_config is None:
        db_config = {
            "host": "162.19.251.55",
            "port": 3306,
            "user": "nidec",
            "password": "MaV38f5xsGQp83",
            "database": "Charges",
        }
    
    try:
        with get_db_connection(db_config) as conn:
            cursor = conn.cursor(dictionary=True)
            try:
                _ensure_table(cursor)

                query = """
                    SELECT * FROM kpi_alertes
                    WHERE detection >= DATE_SUB(NOW(), INTERVAL %s DAY)
                    ORDER BY detection DESC
                """

                cursor.execute(query, (days,))
                results = cursor.fetchall()

                return results
            finally:
                cursor.close()
            
    except Exception as e:
        print(f"Erreur lors de la récupération: {e}")
        return []
