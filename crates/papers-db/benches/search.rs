use arrow_array::{
    FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray, UInt16Array,
    builder::{ListBuilder, StringBuilder},
};
use arrow_schema::Field;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::Rng;
use std::sync::Arc;

use papers_db::schema::{EMBED_DIM, chunks_schema, figures_schema};

fn build_string_list_array(lists: &[Vec<String>]) -> arrow_array::ListArray {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for list in lists {
        for s in list {
            builder.values().append_value(s);
        }
        builder.append(true);
    }
    builder.finish()
}

fn build_vector_array(embeddings: &[Vec<f32>]) -> FixedSizeListArray {
    let flat: Vec<f32> = embeddings.iter().flat_map(|v| v.iter().copied()).collect();
    let flat_array = Arc::new(Float32Array::from(flat));
    let field = Arc::new(Field::new("item", arrow_schema::DataType::Float32, true));
    FixedSizeListArray::new(field, EMBED_DIM, flat_array, None)
}

fn random_vector(rng: &mut impl Rng) -> Vec<f32> {
    (0..EMBED_DIM as usize).map(|_| rng.r#gen::<f32>()).collect()
}

/// Insert `n` synthetic chunks into the store, spread across sections.
async fn insert_synthetic_chunks(store: &papers_db::DbStore, n: usize) {
    let mut rng = rand::thread_rng();
    let schema = chunks_schema();

    let paper_id = "bench-paper";
    let chunks_per_section = 20u16;

    let mut chunk_ids = Vec::with_capacity(n);
    let mut paper_ids = Vec::with_capacity(n);
    let mut embeddings = Vec::with_capacity(n);
    let mut chapter_titles = Vec::with_capacity(n);
    let mut chapter_idxs = Vec::with_capacity(n);
    let mut section_titles = Vec::with_capacity(n);
    let mut section_idxs = Vec::with_capacity(n);
    let mut chunk_idxs = Vec::with_capacity(n);
    let mut depths = Vec::with_capacity(n);
    let mut block_types = Vec::with_capacity(n);
    let mut texts = Vec::with_capacity(n);
    let mut page_starts: Vec<Option<u16>> = Vec::with_capacity(n);
    let mut page_ends: Vec<Option<u16>> = Vec::with_capacity(n);
    let mut titles = Vec::with_capacity(n);
    let mut authors_list = Vec::with_capacity(n);
    let mut years: Vec<Option<u16>> = Vec::with_capacity(n);
    let mut venues: Vec<Option<&str>> = Vec::with_capacity(n);
    let mut tags_list = Vec::with_capacity(n);
    let mut figure_ids_list = Vec::with_capacity(n);

    for i in 0..n {
        let ch = (i / 40) as u16;
        let sec = ((i / chunks_per_section as usize) % 2) as u16;
        let ci = (i % chunks_per_section as usize) as u16;

        chunk_ids.push(format!("{paper_id}/ch{ch}/s{sec}/p{ci}"));
        paper_ids.push(paper_id.to_string());
        embeddings.push(random_vector(&mut rng));
        chapter_titles.push(format!("Chapter {ch}"));
        chapter_idxs.push(ch);
        section_titles.push(format!("Section {sec}"));
        section_idxs.push(sec);
        chunk_idxs.push(ci);
        depths.push("paragraph".to_string());
        block_types.push("text".to_string());
        texts.push(format!("Synthetic chunk text number {i} for benchmarking purposes."));
        page_starts.push(Some(i as u16));
        page_ends.push(Some(i as u16));
        titles.push("Benchmark Paper".to_string());
        authors_list.push(vec!["Author A".to_string()]);
        years.push(Some(2024));
        venues.push(Some("SIGGRAPH"));
        tags_list.push(vec!["bench".to_string()]);
        figure_ids_list.push(vec![]);
    }

    let vectors = build_vector_array(&embeddings);
    let chunk_id_refs: Vec<&str> = chunk_ids.iter().map(|s| s.as_str()).collect();
    let paper_id_refs: Vec<&str> = paper_ids.iter().map(|s| s.as_str()).collect();
    let ch_title_refs: Vec<&str> = chapter_titles.iter().map(|s| s.as_str()).collect();
    let sec_title_refs: Vec<&str> = section_titles.iter().map(|s| s.as_str()).collect();
    let depth_refs: Vec<&str> = depths.iter().map(|s| s.as_str()).collect();
    let bt_refs: Vec<&str> = block_types.iter().map(|s| s.as_str()).collect();
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let title_refs: Vec<&str> = titles.iter().map(|s| s.as_str()).collect();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(chunk_id_refs)),
            Arc::new(StringArray::from(paper_id_refs)),
            Arc::new(vectors),
            Arc::new(StringArray::from(ch_title_refs)),
            Arc::new(UInt16Array::from(chapter_idxs)),
            Arc::new(StringArray::from(sec_title_refs)),
            Arc::new(UInt16Array::from(section_idxs)),
            Arc::new(UInt16Array::from(chunk_idxs)),
            Arc::new(StringArray::from(depth_refs)),
            Arc::new(StringArray::from(bt_refs)),
            Arc::new(StringArray::from(text_refs)),
            Arc::new(UInt16Array::from(page_starts)),
            Arc::new(UInt16Array::from(page_ends)),
            Arc::new(StringArray::from(title_refs)),
            Arc::new(build_string_list_array(&authors_list)),
            Arc::new(UInt16Array::from(years)),
            Arc::new(StringArray::from(venues)),
            Arc::new(build_string_list_array(&tags_list)),
            Arc::new(build_string_list_array(&figure_ids_list)),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let table = store.chunks_table().await.unwrap();
    table.add(Box::new(reader)).execute().await.unwrap();
}

/// Insert `n` synthetic figures into the store.
async fn insert_synthetic_figures(store: &papers_db::DbStore, n: usize) {
    let mut rng = rand::thread_rng();
    let schema = figures_schema();

    let paper_id = "bench-paper";

    let mut figure_ids = Vec::with_capacity(n);
    let mut paper_ids = Vec::with_capacity(n);
    let mut embeddings = Vec::with_capacity(n);
    let mut figure_types = Vec::with_capacity(n);
    let mut captions = Vec::with_capacity(n);
    let mut descriptions = Vec::with_capacity(n);
    let mut image_paths: Vec<Option<&str>> = Vec::with_capacity(n);
    let mut contents: Vec<Option<&str>> = Vec::with_capacity(n);
    let mut pages: Vec<Option<u16>> = Vec::with_capacity(n);
    let mut chapter_idxs = Vec::with_capacity(n);
    let mut section_idxs = Vec::with_capacity(n);
    let mut titles = Vec::with_capacity(n);
    let mut authors_list: Vec<Vec<String>> = Vec::with_capacity(n);
    let mut years: Vec<Option<u16>> = Vec::with_capacity(n);
    let mut venues: Vec<Option<&str>> = Vec::with_capacity(n);
    let mut tags_list: Vec<Vec<String>> = Vec::with_capacity(n);

    for i in 0..n {
        figure_ids.push(format!("{paper_id}/fig{i}"));
        paper_ids.push(paper_id.to_string());
        embeddings.push(random_vector(&mut rng));
        figure_types.push(if i % 3 == 0 { "table" } else { "figure" }.to_string());
        captions.push(format!("Fig. {i}. Synthetic figure caption."));
        descriptions.push(format!("Description of figure {i}."));
        image_paths.push(None);
        contents.push(None);
        pages.push(Some(i as u16));
        chapter_idxs.push((i / 10) as u16);
        section_idxs.push((i % 3) as u16);
        titles.push("Benchmark Paper".to_string());
        authors_list.push(vec![]);
        years.push(Some(2024));
        venues.push(None);
        tags_list.push(vec![]);
    }

    let vectors = build_vector_array(&embeddings);
    let fig_id_refs: Vec<&str> = figure_ids.iter().map(|s| s.as_str()).collect();
    let paper_id_refs: Vec<&str> = paper_ids.iter().map(|s| s.as_str()).collect();
    let ft_refs: Vec<&str> = figure_types.iter().map(|s| s.as_str()).collect();
    let cap_refs: Vec<&str> = captions.iter().map(|s| s.as_str()).collect();
    let desc_refs: Vec<&str> = descriptions.iter().map(|s| s.as_str()).collect();
    let title_refs: Vec<&str> = titles.iter().map(|s| s.as_str()).collect();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(fig_id_refs)),
            Arc::new(StringArray::from(paper_id_refs)),
            Arc::new(vectors),
            Arc::new(StringArray::from(ft_refs)),
            Arc::new(StringArray::from(cap_refs)),
            Arc::new(StringArray::from(desc_refs)),
            Arc::new(StringArray::from(image_paths)),
            Arc::new(StringArray::from(contents)),
            Arc::new(UInt16Array::from(pages)),
            Arc::new(UInt16Array::from(chapter_idxs)),
            Arc::new(UInt16Array::from(section_idxs)),
            Arc::new(StringArray::from(title_refs)),
            Arc::new(build_string_list_array(&authors_list)),
            Arc::new(UInt16Array::from(years)),
            Arc::new(StringArray::from(venues)),
            Arc::new(build_string_list_array(&tags_list)),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let table = store.figures_table().await.unwrap();
    table.add(Box::new(reader)).execute().await.unwrap();
}

fn bench_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("search");
    group.sample_size(20);

    for &n in &[100, 500, 1000] {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("bench.lance").to_string_lossy().into_owned();

        let store = rt.block_on(async {
            let s = papers_db::DbStore::open_for_test(&db_path).await.unwrap();
            insert_synthetic_chunks(&s, n).await;
            s
        });

        let mut rng = rand::thread_rng();
        let query_vec = random_vector(&mut rng);

        group.bench_with_input(BenchmarkId::new("chunks", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let params = papers_db::SearchParams {
                    query: String::new(),
                    paper_ids: None,
                    chapter_idx: None,
                    section_idx: None,
                    filter_year_min: None,
                    filter_year_max: None,
                    filter_venue: None,
                    filter_tags: None,
                    filter_depth: None,
                    limit: 5,
                };
                papers_db::search_with_embedding(&store, params, &query_vec)
            });
        });
    }
    group.finish();
}

fn bench_search_figures(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("search_figures");
    group.sample_size(20);

    for &n in &[100, 500, 1000] {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("bench.lance").to_string_lossy().into_owned();

        let store = rt.block_on(async {
            let s = papers_db::DbStore::open_for_test(&db_path).await.unwrap();
            insert_synthetic_figures(&s, n).await;
            s
        });

        let mut rng = rand::thread_rng();
        let query_vec = random_vector(&mut rng);

        group.bench_with_input(BenchmarkId::new("figures", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let params = papers_db::SearchFiguresParams {
                    query: String::new(),
                    paper_ids: None,
                    filter_figure_type: None,
                    limit: 5,
                };
                papers_db::search_figures_with_embedding(&store, params, &query_vec)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_search, bench_search_figures);
criterion_main!(benches);
