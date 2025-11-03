# analyses/kpi_alertes.py - Script standalone pour remplir la table
import pandas as pd
import mysql.connector
from contextlib import contextmanager
from datetime import datetime
import sys
import os

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tabs.context import get_context

@contextmanager
def get_db_connection():
    """Context manager pour gérer la connexion"""
    conn = None
    try:
        conn = mysql.connector.connect(
            host="162.19.251.55",
            port=3306,
            user="nidec",
            password="MaV38f5xsGQp83",
            database="Charges"
        )
        yield conn
    except mysql.connector.Error as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def save_alerts_to_db(alert_rows):
    """Insère les alertes dans la base de données"""
    if not alert_rows:
        return {"success": True, "rows_affected": 0, "error": None}
    
    try:
        with get_db_connection() as conn:
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
                    row["Détection"],
                    int(row.get("Occurrences sur 12h", 0)),
                    str(row.get("Moment", ""))[:20] if row.get("Moment") else None,
                    str(row.get("EVI Code", ""))[:50] if row.get("EVI Code") else None,
                    str(row.get("Downstream Code PC", ""))[:50] if row.get("Downstream Code PC") else None
                ))
            
            cursor.executemany(insert_query, data)
            conn.commit()
            
            rows_affected = cursor.rowcount
            cursor.close()
            
            return {"success": True, "rows_affected": rows_affected, "error": None}
            
    except Exception as e:
        return {"success": False, "rows_affected": 0, "error": str(e)}


def detect_alerts_from_sess_kpi(sess_kpi, SITE_COL):
    """
    🔥 LIT sess_kpi ET DÉTECTE LES ALERTES
    """
    print(f"\n🔍 Analyse de {len(sess_kpi)} sessions...")
    
    errors_only = sess_kpi[~sess_kpi["is_ok_filt"]].copy()
    
    if errors_only.empty:
        print("✅ Aucune erreur trouvée")
        return []
    
    print(f"⚠️  {len(errors_only)} erreurs détectées")
    
    errors_only["Datetime start"] = pd.to_datetime(errors_only["Datetime start"], errors="coerce")
    errors_only = errors_only.dropna(subset=["Datetime start", "PDC", "type_erreur"])
    errors_only = errors_only.sort_values(["PDC", "type_erreur", "Datetime start"]).reset_index()

    alert_rows = []

    for (pdc, err_type), group in errors_only.groupby(["PDC", "type_erreur"]):
        times = group["Datetime start"].reset_index(drop=True)
        idxs = group["index"].reset_index(drop=True)
        
        processed = set()
        
        for i in range(len(times)):
            if i in processed:
                continue
                
            t0 = times.iloc[i]
            t1 = t0 + pd.Timedelta(hours=12)
            
            window_mask = (times >= t0) & (times <= t1)
            window_indices = times[window_mask].index.tolist()
            
            if len(window_indices) >= 3:
                idx3 = idxs.iloc[i]
                row = sess_kpi.loc[idx3]

                alert_rows.append({
                    "Site": row.get(SITE_COL, "—"),
                    "PDC": pdc,
                    "Type d'erreur": err_type,
                    "Détection": t0,
                    "Occurrences sur 12h": len(window_indices),
                    "Moment": row.get("moment", "—"),
                    "EVI Code": row.get("EVI Error Code", "—"),
                    "Downstream Code PC": row.get("Downstream Code PC", "—")
                })
                
                processed.update(window_indices)
    
    print(f"🚨 {len(alert_rows)} alertes détectées")
    return alert_rows


def main():
    """
    🔥 FONCTION PRINCIPALE - LIT sess_kpi ET REMPLIT LA TABLE
    """
    print("=" * 70)
    print("🚀 DÉTECTION ET SAUVEGARDE DES ALERTES KPI")
    print("=" * 70)
    
    try:
        # 1. Récupérer le contexte (charge sess_kpi)
        print("\n📊 Chargement du contexte...")
        ctx = get_context()
        
        if not hasattr(ctx, 'sess_kpi'):
            print("❌ sess_kpi non trouvé dans le contexte!")
            return
        
        sess_kpi = ctx.sess_kpi
        SITE_COL = getattr(ctx, 'SITE_COL', 'Site')
        
        print(f"✅ Contexte chargé: {len(sess_kpi)} lignes dans sess_kpi")
        
        # 2. Détecter les alertes
        alert_rows = detect_alerts_from_sess_kpi(sess_kpi, SITE_COL)
        
        if not alert_rows:
            print("\n✅ Aucune alerte à sauvegarder")
            return
        
        # 3. Afficher un aperçu
        print(f"\n📋 Aperçu des alertes:")
        for i, alert in enumerate(alert_rows[:5], 1):
            print(f"  {i}. {alert['Site']} | {alert['PDC']} | {alert['Type d\'erreur']} | {alert['Détection']}")
        
        if len(alert_rows) > 5:
            print(f"  ... et {len(alert_rows) - 5} autres")
        
        # 4. Sauvegarder en BDD
        print(f"\n💾 Sauvegarde de {len(alert_rows)} alertes en base de données...")
        result = save_alerts_to_db(alert_rows)
        
        if result["success"]:
            print(f"✅ {result['rows_affected']} alertes sauvegardées avec succès!")
        else:
            print(f"❌ Erreur lors de la sauvegarde: {result['error']}")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ SCRIPT TERMINÉ")
    print("=" * 70)


if __name__ == "__main__":
    main()
