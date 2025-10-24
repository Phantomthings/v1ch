from sqlalchemy import create_engine
import pandas as pd

# 🔧 Connexion à ta base
engine = create_engine("mysql+pymysql://AdminNidec:u6Ehe987XBSXxa4@141.94.31.144:3306/elto")

# 📦 Données MAC <-> Véhicule
data = {
    "MAC": [
        "18:4C:AE:14:34:A1","18:4C:AE:2D:6F:A1","7C:BC:84:42:5D:2F","28:0F:EB:E8:92:33",
        "30:49:50:B1:9E:4A","18:87:00:30:65","A4:53:EE:00:32:55","A4:53:EE:01:5C:E2",
        "18:87:00:92:1F","18:87:02:4F:54","01:6A:C7","28:0F:EB","B0:52:00:DB:D1",
        "B0:52:00:00:03","18:23:AB:D4:B1","18:23:0F:2D:96","18:23:38:C2:EE","7D:FA",
        "7D:FA:09:2A:1E","7D:FA:09:2A:1E","7D:FA:09:97:5D","07:A9:F1","C8:6C:70:5C:AB:0",
        "CC:88:26","18:23:A8:0F:CC","DC:44","FC:A4:7A:1B:0F:47","44:42:2F:01:90:52",
        "8F:21:55","A0:E2:F","BD:F5:64","98:ED:5C:BF:8B:AA","98:ED:5C:E3:D3:52",
        "48:C5:8D:B8:59:73","48:C5:8D:A6:FA:E5","E5:FA:F4","F0:7F:C0:AD:6E:7",
        "F0:7F:C2:69:16:F","80:0A:80:20:66:DF","FC:A4:7A:14:5A:52","FC:A4:7A:15:C2:B2",
        "4E:77:E7:17:C5:C","4E:77:E7:30:62:1","4E:77:E7:15:4C:4","E0:0E:E1:02:81:38",
        "90:12:A1:71:28:1D","90:12:A1:71:CD:F6","90:12:A1:72:FF:01","90:12:A1:00:07:91",
        "F0:7F:C","C0:FB:B6:D3:7D:09","16:81:0E:DC:A4","B0:52:00:DB:D1"
    ],
    "Vehicle": [
        "Renault Mégane e-tech","Renault Mégane e-tech","Renault Zoé ou Renault ou Nissan","Renault Scénic",
        "Dacia Spring","DS 3","Citroën e-C4","Citroën e-Berlingo","Peugeot e-2008","Peugeot e-2008",
        "Peugeot e-208","Peugot e-Riffter","Opel Mokka","Opel e-Corsa","Fiat 500","Fiat 500",
        "Ford e-Mach","VW ID3 / e-Golf","VW ID4 ID5","IDBuzz","Audi Q4 e-tron","VW",
        "Mercedes EQC","Mercedes EQA","Mercedes","Tesla S / 3","Mazda CX30","Tesla","Tesla",
        "Tesla","Tesla Y / 3","Tesla Y / 3","Tesla Y / 3","C40 Volvo","C40 Volvo",
        "XC40 Volvo","MINI SE","Porsche Taycan","MG Marvel","MG ZS EV","MG 4",
        "Kia EV6","Kia EV6","Kia e Niro","Hyundai Kona","Ioniq 6","Ioniq 6","Ioniq 6",
        "Ioniq 5","Ioniq 3 / I3","BYD ATTO 3","RENAULT TRUCKS","LOTUS"
    ]
}

df = pd.DataFrame(data)

df.to_sql(name="mac_lookup", con=engine, schema="indicator", if_exists="replace", index=False)

print("✅ Table 'indicator.mac_lookup' créée et peuplée avec succès.")