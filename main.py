import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import config

emplois_df = pd.read_json(config.EMPLOYEE_DATA_PATH)
formations_df = pd.read_json(config.FORMATION_DATA_PATH)

# Charger le modèle de transformation

model = SentenceTransformer(config.MODEL_NAME)

# Encoder les descriptions de formations

formation_embeddings = model.encode(formations_df['content'].tolist())

def recommend_training(employee_eval):
    """Trouve les formations les plus pertinentes pour une évaluation donnée."""
    eval_embedding = model.encode(employee_eval)
    similarities = util.pytorch_cos_sim(eval_embedding, formation_embeddings)[0]
    top_indices = similarities.argsort(descending=True)[:3]

    recommendations = formations_df.iloc[top_indices.cpu().numpy()]
    return recommendations[['type', 'content', 'source']].to_dict(orient='records')

# Générer les recommandations pour chaque employé

recommendations = []
for index, row in emplois_df.iterrows():
    recs = recommend_training(row['evaluation'])
    recommendations.append({
        'employe': row['employe'],
        'evaluation': row['evaluation'],
        'score': row['score'],
        'suggestions': recs
    })

# Sauvegarder les recommandations

with open('data/suggestions.json', 'w', encoding='utf-8') as f:
    json.dump(recommendations, f, indent=4, ensure_ascii=False)

print("Recommandations générées avec succès.")