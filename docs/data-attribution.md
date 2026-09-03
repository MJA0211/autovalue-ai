# Dataset attribution and trademark notice

AutoValue AI code is MIT-licensed. Dataset licenses and rights are separate and
are not granted by the repository's `LICENSE` file. The project publishes no raw
third-party rows.

## Attributions

### Historical retail asking prices

“US Sales Cars Dataset,” version 2, by Juan Merino Bermejo, distributed through
[Kaggle](https://www.kaggle.com/datasets/juanmerinobermejo/us-sales-cars-dataset).
The dataset metadata displays Apache 2.0. Its documented upstream origin is a
historical Cars.com listing extraction performed by the dataset author.

### Historical wholesale auction sales

“Vehicle Sales Data,” version 1, by Syed Anwar Afridi, distributed through
[Kaggle](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data/data).
The Kaggle page displays an MIT license label.

### Yoad22 Craigslist derivative

“Craigslist Used Cars EDA” by Yoad22, distributed through
[Hugging Face](https://huggingface.co/datasets/Yoad22/craigslist-used-cars-eda)
under the card's declared CC BY 4.0 license. The card identifies the upstream
[Austin Reese Craigslist Cars and Trucks dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data),
whose Kaggle page displays CC0. AutoValue uses Yoad only for controlled offline
experimentation and publishes aggregate results rather than source rows.

### Rebrowser AutoTrader preview

“Rebrowser, AutoTrader Vehicle Listings Dataset (2026),
https://rebrowser.net/products/datasets/autotrader”. The exact free artifact was
audited at Hugging Face revision
`a6cd0c8addded3591ccdfcd6ee4249b454f99792`. The dataset card declares CC BY-NC
4.0 and requires Rebrowser attribution. Use is restricted here to controlled
non-commercial research/educational aggregate analysis.

Rebrowser's terms state that its license does not grant third-party source-IP
rights. The data includes values described as Kelley Blue Book ranges surfaced
through AutoTrader; no reviewed evidence grants AutoValue rights to train or
publish a derivative KBB model. Raw data, premium access, model training, hosted
inference, redistribution, and River updates remain blocked.

## Names and marks

Cars.com, Craigslist, AutoTrader, Kelley Blue Book/KBB, Kaggle, Hugging Face, and
Rebrowser are names or marks of their respective owners. They describe data
provenance only. AutoValue AI is not represented as official, affiliated,
sponsored, or endorsed by those organizations.

See [DATA_SOURCES.md](../DATA_SOURCES.md) and the source-specific records under
[`docs/data-reviews/`](data-reviews/README.md) for the complete permission and
quality boundaries.
