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


def save_alerts_to_db(alert_rows, db_config=None):
    if not alert_rows:
        return {"success": True, "rows_affected": 0, "error": None}
    
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
            
            insert_query = """
                INSERT INTO kpi_alertes 
                (Site, PDC, type_erreur, detection, occurrences_12h, moment, evi_code, downstream_code_pc)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    occurrences_12h = VALUES(occurrences_12h),
                    moment = VALUES(moment),
                    evi_code = VALUES(evi_code),
                    downstream_code_pc = VALUES(downstream_code_pc)
            """
            
            data = []
            for row in alert_rows:
                data.append((
                    str(row.get("Site", ""))[:50],  
                    str(row.get("PDC", ""))[:50],
                    str(row.get("Type d'erreur", ""))[:100],
                    row["Détection"],  # Datetime obligatoire
                    int(row.get("Occurrences sur 12h", 0)),
                    str(row.get("Moment", ""))[:20] if row.get("Moment") else None,
                    str(row.get("EVI Code", ""))[:50] if row.get("EVI Code") else None,
                    str(row.get("Downstream Code PC", ""))[:50] if row.get("Downstream Code PC") else None
                ))
            
            cursor.executemany(insert_query, data)
            conn.commit()
            
            rows_affected = cursor.rowcount
            cursor.close()
            
            return {
                "success": True,
                "rows_affected": rows_affected,
                "error": None
            }
            
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
            
            query = """
                SELECT * FROM kpi_alertes 
                WHERE detection >= DATE_SUB(NOW(), INTERVAL %s DAY)
                ORDER BY detection DESC
            """
            
            cursor.execute(query, (days,))
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
    except Exception as e:
        print(f"Erreur lors de la récupération: {e}")
        return []
