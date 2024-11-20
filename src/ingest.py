import os
import pandas as pd

import minsearch


DATA_PATH = os.getenv("DATA_PATH", "../data/cleaned_data.csv")


def load_index(data_path=DATA_PATH):
    df = pd.read_csv(data_path)

    documents = df.to_dict(orient="records")

    index = minsearch.Index(
        text_fields=[
            'type_d_article', 
            'numéro_de_l_article_ou_de_la_loi',
            'description_ou_texte_complet', 
            'mots_clés_ou_sujets_abordés',
            'date_de_publication', 'source'
        ],
        keyword_fields=["id"],
    )

    index.fit(documents)
    return index