from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class GridPos(BaseModel):
    x: int
    y: int
    w: int
    h: int

class Datasource(BaseModel):
    type: str = "postgres"
    uid: str = "aeo8prusu1i4gc"

class Target(BaseModel):
    refId: str = "A"
    format: str  # "time_series" or "table"
    rawSql: str

class Tooltip(BaseModel):
    mode: str = "single"

class Legend(BaseModel):
    displayMode: str = "list"
    placement: str = "bottom"

class ReduceOptions(BaseModel):
    calcs: List[str] = ["lastNotNull"]
    fields: str = ""
    values: bool = True

class PanelOptions(BaseModel):
    tooltip: Tooltip = Tooltip()
    legend: Legend = Legend()
    reduceOptions: ReduceOptions = ReduceOptions()


class Panel(BaseModel):
    id: int
    type: str  # "barchart", "piechart", "table", "timeseries", "stat"
    title: str
    datasource: Datasource = Datasource()
    targets: List[Target]
    gridPos: GridPos
    options: PanelOptions = PanelOptions()

class TimeRange(BaseModel):
    from_time: str = Field(alias="from", default="now-5y")
    to: str = "now"

class TimePicker(BaseModel):
    refresh_intervals: List[str] = ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h", "2h", "1d"]
    time_options: List[str] = ["5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d", "30d", "90d", "6M", "1y", "5y"]

class DashboardContent(BaseModel):
    title: str
    schemaVersion: int = 36
    version: int = 1
    refresh: str = "5s"
    time: TimeRange = TimeRange()
    timepicker: TimePicker = TimePicker()
    panels: List[Panel]

class DashboardSpec(BaseModel):
    dashboard: DashboardContent
    overwrite: bool = False

    class Config:
        allow_population_by_field_name = True