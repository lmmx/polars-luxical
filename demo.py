"""Demo script for polars-luxical: document similarity and near-duplicate detection."""

import polars as pl
from polars_luxical import register_model

model_id = "datologyai/luxical-one"
register_model(model_id)  # downloads and caches the model on first use

article = (
    "Polars is a DataFrame library written in Rust. It uses Apache Arrow as its "
    "memory model and supports lazy evaluation, query optimisation and parallel "
    "execution, so pipelines written with its expression API use every core."
)
df = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "text": [
            article,
            (
                "Sourdough needs flour, water, salt and a lively starter. Mix, fold, "
                "prove overnight in the fridge and bake in a very hot oven."
            ),
            # A lightly edited copy of the first article, as found across web crawls
            "Home | Blog | Contact\n" + article.replace("Rust", "the Rust language"),
            (
                "The 1966 World Cup final was played at Wembley Stadium, where England "
                "beat West Germany 4-2 after extra time."
            ),
            (
                "Bread baking at home: combine flour, water and yeast, knead, let the "
                "dough rise, shape it and bake until the crust is golden."
            ),
        ],
    },
)

# Embed every document
df = df.luxical.embed(columns="text", model_name=model_id)

# Near-duplicates: row 3 is flagged as a copy of row 1
marked = df.luxical.mark_duplicates()
print(marked.select("id", "duplicate_of", "duplicate_of_similarity", "text"))

# Keep the first of each group of near-duplicates
print(df.luxical.dedup().select("id", "text"))

# Query by example: use a document (or several seed documents) as the query.
# Luxical compares documents with documents; it was not trained on short queries.
similar = df.luxical.retrieve(
    query=["Homemade bread recipe: flour, water, yeast and a hot oven."],
    model_name=model_id,
    k=2,
)
print(similar.select("id", "similarity", "text"))
