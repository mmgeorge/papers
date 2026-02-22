use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

pub const EMBED_DIM: i32 = 768;

fn string_list_field(name: &str) -> Field {
    Field::new(
        name,
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        false,
    )
}

fn string_list_field_nullable(name: &str) -> Field {
    Field::new(
        name,
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        true,
    )
}

fn vector_field() -> Field {
    Field::new(
        "vector",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            EMBED_DIM,
        ),
        false,
    )
}

pub fn chunks_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("chunk_id", DataType::Utf8, false),
        Field::new("paper_id", DataType::Utf8, false),
        vector_field(),
        Field::new("chapter_title", DataType::Utf8, false),
        Field::new("chapter_idx", DataType::UInt16, false),
        Field::new("section_title", DataType::Utf8, false),
        Field::new("section_idx", DataType::UInt16, false),
        Field::new("chunk_idx", DataType::UInt16, false),
        Field::new("depth", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
        Field::new("page_start", DataType::UInt16, true),
        Field::new("page_end", DataType::UInt16, true),
        Field::new("title", DataType::Utf8, false),
        string_list_field("authors"),
        Field::new("year", DataType::UInt16, true),
        Field::new("venue", DataType::Utf8, true),
        string_list_field("tags"),
        string_list_field("figure_ids"),
    ]))
}

pub fn figures_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("figure_id", DataType::Utf8, false),
        Field::new("paper_id", DataType::Utf8, false),
        vector_field(),
        Field::new("figure_type", DataType::Utf8, false),
        Field::new("caption", DataType::Utf8, false),
        Field::new("description", DataType::Utf8, false),
        Field::new("image_path", DataType::Utf8, true),
        Field::new("page", DataType::UInt16, true),
        Field::new("chapter_idx", DataType::UInt16, false),
        Field::new("section_idx", DataType::UInt16, false),
        Field::new("title", DataType::Utf8, false),
        string_list_field_nullable("authors"),
        Field::new("year", DataType::UInt16, true),
        Field::new("venue", DataType::Utf8, true),
        string_list_field_nullable("tags"),
    ]))
}
