//! The subset of W&B's run-record protobufs that an offline run needs,
//! hand-written as prost messages. Field numbers follow
//! `wandb/proto/wandb_internal.proto` and `wandb_base.proto` in
//! github.com/wandb/wandb (MIT licence). Records and fields not listed here
//! (telemetry, output, files, stats, ...) are skipped as unknown fields when
//! decoding.

#[derive(Clone, PartialEq, prost::Message)]
pub struct Record {
    #[prost(int64, tag = "1")]
    pub num: i64,
    #[prost(oneof = "RecordType", tags = "2, 3, 5, 17, 18, 20, 21")]
    pub record_type: Option<RecordType>,
    #[prost(string, tag = "19")]
    pub uuid: String,
    #[prost(message, optional, tag = "200")]
    pub info: Option<RecordInfo>,
}

#[derive(Clone, PartialEq, prost::Oneof)]
pub enum RecordType {
    #[prost(message, tag = "2")]
    History(HistoryRecord),
    #[prost(message, tag = "3")]
    Summary(SummaryRecord),
    #[prost(message, tag = "5")]
    Config(ConfigRecord),
    #[prost(message, tag = "17")]
    Run(RunRecord),
    #[prost(message, tag = "18")]
    Exit(RunExitRecord),
    #[prost(message, tag = "20")]
    Final(FinalRecord),
    #[prost(message, tag = "21")]
    Header(HeaderRecord),
}

/// `_RecordInfo` (wandb_base.proto).
#[derive(Clone, PartialEq, prost::Message)]
pub struct RecordInfo {
    #[prost(string, tag = "1")]
    pub stream_id: String,
}

/// `google.protobuf.Timestamp`.
#[derive(Clone, PartialEq, prost::Message)]
pub struct Timestamp {
    #[prost(int64, tag = "1")]
    pub seconds: i64,
    #[prost(int32, tag = "2")]
    pub nanos: i32,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct RunRecord {
    #[prost(string, tag = "1")]
    pub run_id: String,
    #[prost(string, tag = "2")]
    pub entity: String,
    #[prost(string, tag = "3")]
    pub project: String,
    #[prost(message, optional, tag = "4")]
    pub config: Option<ConfigRecord>,
    #[prost(message, optional, tag = "5")]
    pub summary: Option<SummaryRecord>,
    #[prost(string, tag = "6")]
    pub run_group: String,
    #[prost(string, tag = "7")]
    pub job_type: String,
    #[prost(string, tag = "8")]
    pub display_name: String,
    #[prost(string, tag = "9")]
    pub notes: String,
    #[prost(string, repeated, tag = "10")]
    pub tags: Vec<String>,
    #[prost(string, tag = "13")]
    pub host: String,
    #[prost(int64, tag = "14")]
    pub starting_step: i64,
    #[prost(message, optional, tag = "17")]
    pub start_time: Option<Timestamp>,
    #[prost(int32, tag = "20")]
    pub runtime: i32,
    #[prost(message, optional, tag = "200")]
    pub info: Option<RecordInfo>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct HistoryRecord {
    #[prost(message, repeated, tag = "1")]
    pub item: Vec<HistoryItem>,
    #[prost(message, optional, tag = "2")]
    pub step: Option<HistoryStep>,
    #[prost(message, optional, tag = "200")]
    pub info: Option<RecordInfo>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct HistoryItem {
    #[prost(string, tag = "1")]
    pub key: String,
    #[prost(string, repeated, tag = "2")]
    pub nested_key: Vec<String>,
    /// Newer producers may set a typed value instead of `value_json`.
    #[prost(oneof = "HistoryValue", tags = "3, 4, 5, 6, 7, 8")]
    pub value: Option<HistoryValue>,
    #[prost(string, tag = "16")]
    pub value_json: String,
}

#[derive(Clone, PartialEq, prost::Oneof)]
pub enum HistoryValue {
    /// `google.protobuf.NullValue` (an enum; always 0).
    #[prost(int32, tag = "3")]
    None(i32),
    #[prost(bool, tag = "4")]
    Boolean(bool),
    #[prost(sint64, tag = "5")]
    Integer(i64),
    #[prost(double, tag = "6")]
    Number(f64),
    #[prost(string, tag = "7")]
    Text(String),
    #[prost(string, tag = "8")]
    Json(String),
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct HistoryStep {
    #[prost(int64, tag = "1")]
    pub num: i64,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct SummaryRecord {
    #[prost(message, repeated, tag = "1")]
    pub update: Vec<SummaryItem>,
    #[prost(message, repeated, tag = "2")]
    pub remove: Vec<SummaryItem>,
    #[prost(message, optional, tag = "200")]
    pub info: Option<RecordInfo>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct SummaryItem {
    #[prost(string, tag = "1")]
    pub key: String,
    #[prost(string, repeated, tag = "2")]
    pub nested_key: Vec<String>,
    #[prost(string, tag = "16")]
    pub value_json: String,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct ConfigRecord {
    #[prost(message, repeated, tag = "1")]
    pub update: Vec<ConfigItem>,
    #[prost(message, repeated, tag = "2")]
    pub remove: Vec<ConfigItem>,
    #[prost(message, optional, tag = "200")]
    pub info: Option<RecordInfo>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct ConfigItem {
    #[prost(string, tag = "1")]
    pub key: String,
    #[prost(string, repeated, tag = "2")]
    pub nested_key: Vec<String>,
    #[prost(string, tag = "16")]
    pub value_json: String,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct RunExitRecord {
    #[prost(int32, tag = "1")]
    pub exit_code: i32,
    #[prost(int32, tag = "2")]
    pub runtime: i32,
    #[prost(bool, tag = "3")]
    pub not_complete: bool,
    #[prost(message, optional, tag = "200")]
    pub info: Option<RecordInfo>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct FinalRecord {
    #[prost(message, optional, tag = "200")]
    pub info: Option<RecordInfo>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct HeaderRecord {
    #[prost(message, optional, tag = "1")]
    pub version_info: Option<VersionInfo>,
    #[prost(message, optional, tag = "200")]
    pub info: Option<RecordInfo>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct VersionInfo {
    #[prost(string, tag = "1")]
    pub producer: String,
    #[prost(string, tag = "2")]
    pub min_consumer: String,
}
