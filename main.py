# 1. 표준 라이브러리 & 데이터 핸들링
import random
import psycopg2
import pandas as pd
import plotly.express as px
import requests
from collections import defaultdict

# 1-1. 셔틀 정보 전처리 & 지표 계산
from shuttle_info import transform_cur_path, compute_shuttle_metrics

# 2. Dash & Plotly
import dash
from dash import dcc, html, callback_context, dash_table, no_update
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dash_table import DataTable

# 프로젝트 루트 모듈
from shuttle_info import transform_cur_path, compute_shuttle_metrics
import data_utils

# 4. KPI 유틸
from kpi_utils import (
    count_running_shuttles,
    compute_avg_occupancy,
    compute_rejection_rate
)

import os 
# 1) 이 파일이 있는 디렉토리 (프로젝트 루트)
BASE_DIR = os.path.dirname(__file__)
# 2) assets 폴더 경로
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

import base64
from io import BytesIO
from PIL import Image

#####
import data_utils
from Environment.EnvironmentLoader import EnvironmentLoader
node_data=EnvironmentLoader("./JSON/",["map_graph_with_vectors", "passengerInfo", "shuttleInfo", "setup"]).getConfiguration().getConfiguration("node_data")
def find_nearest_nodes(x, y,nodeInfo, num_neighbors=1):
        x=(x-126)*10000
        y=(y-37)*10000
        distances = []
        for node_id, coordinates  in nodeInfo.items():
            node_x, node_y = coordinates 
            distance = ((x - node_x) ** 2 + (y - node_y) ** 2) ** 0.5
            distances.append((distance, node_id))
        distances.sort()
        nearest_nodes = [node_id for _, node_id in distances[:num_neighbors]]
        return nearest_nodes


#####

# ──────────────────────────────────────────────────────────────
# 설정: 전체 셔틀 수
# ──────────────────────────────────────────────────────────────
global TOTAL_SHUTTLES
TOTAL_SHUTTLES = 8

# ──────────────────────────────────────────────────────────────
# 유틸리티: 색상 할당 & CSV 읽기
# ──────────────────────────────────────────────────────────────
available_colors = px.colors.qualitative.Plotly
shuttle_colors = {}
# ──────────────────────────────────────────────────────────────
# 3) 셔틀 ID → 고정 색 아이콘 파일 매핑
#    assets 폴더에 passenger_636EFA.png 등 미리 만들어 두셔야 합니다.
passenger_icon_map = {
    "SHUTTLE0001": "/assets/passenger_636EFA.png",
    "SHUTTLE0002": "/assets/passenger_EF553B.png",
    "SHUTTLE0003": "/assets/passenger_00CC96.png",
    "SHUTTLE0004": "/assets/passenger_AB63FA.png",
    "SHUTTLE0005": "/assets/passenger_FFA15A.png",
    "SHUTTLE0006": "/assets/passenger_19D3F3.png",
    "SHUTTLE0007": "/assets/passenger_FF6692.png",
    "SHUTTLE0008": "/assets/passenger_B6E880.png",
    # (필요한 만큼 더 추가)
}
# 배차 전 기본 아이콘
default_passenger_icon = "/assets/passenger_gray.png"
# ──────────────────────────────────────────────────────────────

def assign_shuttle_color(sid):
    if sid not in shuttle_colors:
        if len(shuttle_colors) < len(available_colors):
            shuttle_colors[sid] = available_colors[len(shuttle_colors)]
        else:
            while True:
                c = f"#{random.randint(0,0xFFFFFF):06x}"
                if c not in shuttle_colors.values():
                    shuttle_colors[sid] = c
                    break
    return shuttle_colors[sid]

def read_csv_with_fallback(path):
    for enc in ['utf-8','cp949','euc-kr']:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"모든 인코딩 시도 실패: {path}")

mapping_df = read_csv_with_fallback(
    os.path.join(ASSETS_DIR, "mapping.csv")
)
# 컬럼명이 'id' 와 'NODE_NAME' 이므로, 그대로 사용합니다.
node_name_map = mapping_df.set_index("id")["NODE_NAME"].to_dict()
# ──────────────────────────────────────────────────────────────
# 지도 좌표 & 노드 이름 매핑 로드
# ──────────────────────────────────────────────────────────────
df_map   = read_csv_with_fallback(os.path.join(ASSETS_DIR, "위도경도_바인딩.csv"))
df_nodes = read_csv_with_fallback(os.path.join(ASSETS_DIR, "cor_node.csv"))
df_links = read_csv_with_fallback(os.path.join(ASSETS_DIR, "cor_link.csv"))

latlon_dict = (
    df_map
      .rename(columns={"위도":"lat","경도":"lon"})
      .set_index("id")[["lat","lon"]]
      .to_dict("index")
)

df_nodes = df_nodes.rename(columns={"NODE_NAME":"node_name","x":"lon","y":"lat"})
coord_to_name = {
    (row.lon, row.lat): row.node_name
    for _, row in df_nodes.iterrows()
}


# ──────────────────────────────────────────────────────────────
# 전역 상태 변수
# ──────────────────────────────────────────────────────────────
global all_rows
all_rows = []
global last_loaded_time
last_loaded_time = 0.0
global current_index
current_index = 0
global all_passenger_rows
all_passenger_rows = []
global last_loaded_pass_time
last_loaded_pass_time = 0.0
global latest_time
latest_time = 0.0

global shuttle_paths
shuttle_paths = {}  # sid → [{lon,lat,node_name}, …]
global current_shuttles
current_shuttles = {}  # sid → (lon,lat,occupancy)

# ──────────────────────────────────────────────────────────────
# DB 로우 로드 함수
# ──────────────────────────────────────────────────────────────
def load_new_db_rows():
    global all_rows, last_loaded_time
    cur.execute("""
       SELECT
            -- 문자열로 받을 필드들 (원래 TEXT 컬럼)
            scenario_info     AS scenario_info,   -- TEXT
            shuttle_id        AS shuttle_id,      -- TEXT
            shuttle_state     AS shuttle_state,   -- TEXT
            cur_dst           AS cur_dst,         -- TEXT
            cur_path          AS cur_path,        -- TEXT
            cur_psgr          AS cur_psgr,        -- TEXT
            cur_node          AS cur_node,        -- TEXT

            -- 숫자로 받을 필드들
            currenttime::DOUBLE PRECISION  AS currenttime,   -- float
            cur_psgr_num::DOUBLE PRECISION AS cur_psgr_num   -- float
        FROM vehicle_kpi
        WHERE (currenttime::DOUBLE PRECISION) > %s
        ORDER BY (currenttime::DOUBLE PRECISION) ASC
        LIMIT 5000
    """, (last_loaded_time,))
    rows = cur.fetchall()
    if not rows:
        return
    cols = [d[0] for d in cur.description]
    for r in rows:
        rec = dict(zip(cols, r))
        if rec["currenttime"] is None:
            continue
        rec["path_nodes"] = transform_cur_path(
            rec.get("cur_path",""),
            latlon_dict,
            coord_to_name
        )
        all_rows.append(rec)
    all_rows.sort(key=lambda r: r["currenttime"])
    last_loaded_time = all_rows[-1]["currenttime"]

def load_new_passenger_rows():
    global all_passenger_rows, last_loaded_pass_time

    # 1) 새로 추가된 행(calltime > last_loaded_pass_time)
    #    + 기존 대기 중이던 승객 중에서 success가 NULL→Not NULL 로 바뀐 행도 함께 조회
    cur.execute("""
        SELECT
          passenger_id::BIGINT            AS passenger_id,       -- int
          psgrnum::BIGINT                 AS psgrnum,            -- int
          arr_node::BIGINT                AS arr_node,           -- int
          calltime::DOUBLE PRECISION      AS calltime,           -- float
          waitstarttime::BIGINT           AS waitstarttime,      -- int
          dep_node                        AS dep_node,           -- str
          dep_node_expanded               AS dep_node_expanded,  -- str
          shuttleid                       AS shuttleid,          -- str
          boardingtime::DOUBLE PRECISION  AS boardingtime,       -- float
          expectedwaitingtime::DOUBLE PRECISION AS expectedwaitingtime, -- float
          expectedarrivaltime::DOUBLE PRECISION AS expectedarrivaltime, -- float
          arrivaltime::DOUBLE PRECISION   AS arrivaltime,        -- float
          success::Boolean                         AS success             -- str (NULL 포함)
        FROM passengers_kpi
       WHERE (calltime::DOUBLE PRECISION) > %s
          OR (
               (calltime::DOUBLE PRECISION) <= %s
               AND success IS NOT NULL
             )
       ORDER BY (calltime::DOUBLE PRECISION) ASC
    """
    , (last_loaded_pass_time, last_loaded_pass_time))

    rows = cur.fetchall()
    if not rows:
        return

    cols = [desc[0] for desc in cur.description]

    # 2) 기존 리스트에 있는 레코드 인덱스를 key로 매핑
    existing = {
        (r["passenger_id"], r["calltime"]): idx
        for idx, r in enumerate(all_passenger_rows)
    }

    # 3) 페칭한 각 행을 신규/업데이트로 분기
    for r in rows:
        rec = dict(zip(cols, r))
        key = (rec["passenger_id"], rec["calltime"])
        if key in existing:
            # 이미 있던 레코드: success 등 필드를 업데이트
            all_passenger_rows[ existing[key] ].update(rec)
        else:
            # 새로운 호출 기록: 리스트에 추가
            all_passenger_rows.append(rec)

    # 4) calltime 순으로 재정렬
    all_passenger_rows.sort(key=lambda x: x["calltime"])

    # 5) last_loaded_pass_time 갱신: 가장 큰 calltime
    last_loaded_pass_time = max(r["calltime"] for r in all_passenger_rows)


def create_stats_box(stats_cards):
    """
    실시간 현황판을 담은 카드 컴포넌트 반환
    stats_cards: 기존에 정의하신 4개의 KPI 카드(운영 셔틀/평균 탑승객/누적 탑승객/거절률)를 담은 Div
    """
    return dbc.Card(
        [
            html.H4(
                "실시간 현황",
                style={
                    "fontWeight": "bold",
                    "textAlign": "center",
                    "marginBottom": "1rem"
                }
            ),
            stats_cards
        ],
        style={
            "borderRadius": "12px",
            "backgroundColor": "#fff",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
            "padding": "1rem",
            "marginBottom": "1rem"
        }
    )

def get_psgrnum_for_shuttle(sid):
    """
    주어진 셔틀 ID에 대응되는 승객들의 승객 수(psgrnum)를 합산하여 반환합니다.
    """
    total_psgrnum = 0
    for passenger in all_passenger_rows:
        if passenger["passenger_id"] == sid:
            total_psgrnum = passenger.get("psgrnum", 1)  # 기본값 1로 설정
            return total_psgrnum
    return 0



    
# ──────────────────────────────────────────────────────────────
# DB 연결
# ──────────────────────────────────────────────────────────────


########### <원격 연결> #########
conn = psycopg2.connect(
    host='172.17.98.49',   # 원격 서버 IP
    port=5432,
    dbname='test_drt',    # 기존 DB 이름
    user='t_d',           # 생성·권한 부여한 사용자
    password='0330'       # t_d 계정 비밀번호
)
conn.autocommit = True
cur = conn.cursor()
##################################


############### <local 연결 > ################
#DB = dict(host="localhost", database="postgres",
#          user="postgres", password="0123456789")
#conn = psycopg2.connect(**DB)
#conn.autocommit = True
#cur = conn.cursor()
############################################





# ──────────────────────────────────────────────────────────────
# Dash 앱 & 레이아웃 (멀티페이지)
# ──────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.COSMO],
    suppress_callback_exceptions=True
)

# ▶ 대시보드 전용 레이아웃
# (원본 그대로 복사)
stats_cards = html.Div(
    style={
        "display": "flex",
        "gap": "1rem",
        "marginBottom": "1rem",
        "justifyContent": "center"
    },
    children=[
        dbc.Card(
            dbc.CardBody([
                html.H6([
                        "운영 셔틀 /",  # 첫 줄
                        html.Br(),
                        "전체 셔틀"   # 둘째 줄
                    ],
                 style={"fontWeight": "600"}),
                html.H5(id="total-shuttles", style={"fontWeight": "bold", "color": "BRIGHT_BLUE"}),
            ]),
            style={"flex": "1", "borderRadius": "12px"}
        ),
        dbc.Card(
            dbc.CardBody([
                html.H6("평균 탑승객", style={"fontWeight": "600"}),
                html.H5(id="avg-occupancy", style={"fontWeight": "bold", "color": "BRIGHT_BLUE"}),
            ]),
            style={"flex": "1", "borderRadius": "12px"}
        ),
        dbc.Card(
            dbc.CardBody([
                html.H6("누적 탑승객", style={"fontWeight": "600"}),
                html.H5(id="cumulative_passenger_num", style={"fontWeight": "bold", "color": "BRIGHT_BLUE"}),
            ]),
            style={"flex": "1", "borderRadius": "12px"}
        ),
        dbc.Card(
            dbc.CardBody([
                html.H6("거절률", style={"fontWeight": "600"}),
                html.H5(id="rejection-rate", style={"fontWeight": "bold", "color": "BRIGHT_BLUE"}),
            ]),
            style={"flex": "1", "borderRadius": "12px"}
        ),
    ]
)


analysis_subtabs = dbc.Tabs(
    [dbc.Tab(label="거절률 높은 지역",   tab_id="tab-rej-area"),
     dbc.Tab(label="대기시간 오차",      tab_id="tab-rej-rank"),
     dbc.Tab(label="이동시간 오차",      tab_id="tab-veh-stats")],
    id="analysis-tabs", active_tab="tab-rej-area"
)
left_panel = html.Div(
    style={"flex":"0 0 60%","position":"relative","padding":"10px"},
    children=[
        dcc.Graph(id="map-graph", style={"width":"100%","height":"100%"}),
        dbc.Switch(id="toggle-density", value=False,
                   style={"position":"absolute","top":"10px","left":"10px","zIndex":1001}),
        html.Div("밀집도", style={"position":"absolute","top":"10px","left":"50px","padding":"2px 6px","background":"rgba(0,0,0,0.4)","color":"#fff","borderRadius":"4px","zIndex":1001}),
        dbc.Switch(id="toggle-reject-heatmap", value=False,
                   style={"position":"absolute","top":"40px","left":"10px","zIndex":1001}),
        html.Div("거절 히트맵", style={"position":"absolute","top":"40px","left":"50px","padding":"2px 6px","background":"rgba(0,0,0,0.4)","color":"#fff","borderRadius":"4px","zIndex":1001}),
        html.Div(id="current-time-display", style={"position":"absolute","top":"70px","left":"10px","padding":"6px","background":"rgba(0,0,0,0.6)","color":"#fff","borderRadius":"4px","zIndex":1000}),
        html.Div(id="shuttle-info-card", children=[html.Button("",id="close-shuttle-btn",n_clicks=0,style={"display":"none"})], style={"position":"absolute","top":"100px","left":"10px","zIndex":1002,"width":"280px"}),
    ]
)
# 기존 right_panel 정의를 아래처럼 바꿔주세요.

right_panel = html.Div(
    style={"flex":"0 0 40%", "padding":"10px","overflow":"auto"},
    children=[
        # ───────────────── stats 카드(변경 없음)
        create_stats_box(stats_cards),

        # ───────────────── 운영 분석 지표 패널
        dbc.Card(
            [
                html.H4(
                    "운영 분석 지표",
                    style={
                        "fontWeight": "bold",
                        "textAlign": "center",
                        "marginBottom": "1rem"
                    }
                ),
                # 기존에 analysis_subtabs 로 정의된 탭들
                analysis_subtabs,
                # tab-content 콜백이 여기에 렌더링됩니다.
                html.Div(
                    id="tab-content",
                    style={"padding": "1rem", "background": "#f9f9f9"}
                )
            ],
            style={
                "borderRadius": "12px",
                "backgroundColor": "#fff",
                "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
                "padding": "0",
                "marginBottom": "1rem"
            }
        ),
    ]
)

visual_layout = html.Div(style={"display":"flex","height":"100vh"}, children=[left_panel, right_panel])


# ──────────────────────────────────────────────────────────────
# 1) 메인 레이아웃
# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
# 1) 메인 레이아웃
# ──────────────────────────────────────────────────────────────
# … (중략) …

# 1) 메인 레이아웃
main_layout = html.Div([
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("관리자 대시보드", href="/"),
            dbc.Button("데이터 보기", href="/data", color="light", className="ms-auto")
        ]),
        style={"backgroundColor": "#0e4a84"},
        dark=True,
        className="mb-4"
    ),
    visual_layout,
    dcc.Interval(id="interval", interval=1000, n_intervals=0),
    dcc.Store(id="show-density", data=False),
    dcc.Store(id="show-reject-heatmap", data=False),
    dcc.Store(id="chat-open", data=False),

    # 2) FAB 버튼 ── 인라인 스타일 제거, ID만 남김
    html.Button(
        "🗨️",
        id="open-chat-btn"
        # style={…} 부분 모두 제거했습니다.
    ),

    # 3) 챗봇 패널 ── style 대신 className="" 만 지정
    html.Div(
        id="chat-panel",
        className="",   # 초기 상태: “open” 클래스가 붙어 있지 않음 → CSS에서 display: none 처리
        children=[
            # (3-1) 닫기 버튼, 인라인 스타일 제거
            html.Button("×", id="close-chat-btn"),

            # (3-2) 대화 내역, 인라인 스타일 제거
            html.Div(id="chat-output"),

            # (3-3) 입력창
            dcc.Textarea(
                id="chat-input",
                placeholder="셔틀 3은 지금 어디에 있어?"
            ),

            # (3-4) 전송 버튼
            html.Div(
                html.Button("Send", id="chat-send"),
                style={"textAlign": "right", "marginTop": "0.5rem"}
                # 버튼 내부만 margin 주기 위해 여전히 간단한 인라인 스타일을 남겼습니다.
            )
        ]
    )
])



# ──────────────────────────────────────────────────────────────
# 2) 데이터 보기 레이아웃
# ──────────────────────────────────────────────────────────────
data_layout = html.Div([
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("데이터 보기", href="/data"),
            dbc.Button("대시보드로 이동", href="/", color="light", className="ms-auto")
        ]),
        style={"backgroundColor": "#0e4a84"},dark=True
    ),
    html.Div([
        # 필터 컨트롤
        html.Div([
            html.Label("승객 ID:"), dcc.Input(id="filter-passenger-id", type="number", placeholder="ID 입력"),
            html.Label("성공 여부:"), dcc.Dropdown(id="filter-success",
                 options=[{"label":"성공","value":True},{"label":"실패","value":False}],
                 multi=True, placeholder="선택", style={"width":"150px"}),
            html.Label("셔틀 ID:"), dcc.Input(id="filter-shuttle-id",type="text",placeholder="S1")
        ], style={"display":"flex","gap":"1rem","padding":"1rem","alignItems":"center"}),
        # 테이블
        html.H5("Passengers KPI"),
        dash_table.DataTable(
            id="table-passengers",
            columns=[
                {"name":"Passenger ID",        "id":"passenger_id"},
                {"name":"Call Time",           "id":"calltime"},
                {"name":"Dep Node",            "id":"dep_node_expanded"},
                {"name":"Arr Node",            "id":"arr_node"},
                {"name":"Shuttle ID",          "id":"shuttleid"},
                {"name":"Success",             "id":"success"},
                {"name":"Wait Start Time",     "id":"waitstarttime"},
                {"name":"Boarding Time",       "id":"boardingtime"},
                {"name":"Expected Arrival",    "id":"expectedarrivaltime"},
                {"name":"Expected Waiting",    "id":"expectedwaitingtime"},
                {"name":"Arrival Time",        "id":"arrivaltime"},
             ],
            page_size=20, filter_action="native", sort_action="native",
            sort_by=[{"column_id":"calltime","direction":"asc"}],
            style_table={"overflowX":"auto"}
        ),
        html.H5("Vehicle KPI", className="mt-4"),
        dash_table.DataTable(
            id="table-vehicles",
            columns=[{"name":c,"id":c} for c in ["currenttime","shuttle_id","cur_node","cur_psgr_num"]],
            page_size=20, filter_action="native", sort_action="native",
            sort_by=[{"column_id":"currenttime","direction":"asc"}],
            style_table={"overflowX":"auto"}
        )
    ])
])

# ──────────────────────────────────────────────────────────────
# 3) 전체 앱 레이아웃 (라우팅)
# ──────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

# ──────────────────────────────────────────────────────────────
# 페이지 라우팅 콜백
# ──────────────────────────────────────────────────────────────
@app.callback(
    Output("page-content","children"),
    Input("url","pathname")
)
def display_page(pathname):
    if pathname == "/data":
        return data_layout
    return main_layout

# ──────────────────────────────────────────────────────────────
# 이하 기존 콜백들 (수정 없이 그대로)
# 1) Primary Tabs 렌더링
# open / close 버튼 누를 때 chat-open 토글
@app.callback(
    Output("chat-open", "data"),
    [Input("open-chat-btn", "n_clicks"),
     Input("close-chat-btn", "n_clicks")],
    prevent_initial_call=True
)
def toggle_chat(open_clicks, close_clicks):
    ctx = callback_context.triggered_id
    return True if ctx == "open-chat-btn" else False

# chat-open 값에 따라 패널 보이기/숨기기
@app.callback(
    Output("chat-panel", "className"),
    Input("chat-open", "data"),
    prevent_initial_call=True
)
def show_hide_chat(opened):
    # opened == True  → "open" 클래스를 붙여 줘서 CSS #chat-panel.open { display:flex } 적용
    # opened == False → 빈 문자열 → display:none 이 CSS에서 처리
    return "open" if opened else ""





@app.callback(
    Output("tab-content", "children"),
    Input("analysis-tabs", "active_tab")
)
def render_tab(active_tab):
    if active_tab == "tab-rej-area":
        return DataTable(
            id="rej-area-table",
            columns=[
                {"name":"지역",     "id":"region"},
                {"name":"거절 횟수","id":"count"}
            ],
            data=[],              # 초기에는 빈 리스트
            page_size=10,
            sort_action="native",
            style_cell={"textAlign":"center"},
            style_header={"fontWeight":"bold"}
        )  
    elif active_tab == "tab-rej-rank":
        return dcc.Graph(id="wait-error-graph")
    else:
        return dcc.Graph(id="travel-error-graph")


# 3) 토글 스토어 업데이트
@app.callback(Output("show-density","data"), Input("toggle-density","value"))
def toggle_density(on): return on
@app.callback(Output("show-reject-heatmap","data"), Input("toggle-reject-heatmap","value"))
def toggle_reject_heatmap(on): return on

# 4) DB → 메모리 로드
@app.callback(Output("interval","n_intervals"), Input("interval","n_intervals"))
def update_sim(n):
    global current_index, latest_time
    load_new_db_rows(); load_new_passenger_rows()
    for row in all_rows[current_index:]:
        sid = row["shuttle_id"]
        occ = row.get("cur_psgr_num",0) or 0 
        pts = row.get("path_nodes",[])
        shuttle_paths[sid] = pts
        if len(pts)<=1: occ=0
        try:
            nid=int(row["cur_node"])
            coord=latlon_dict.get(nid)
            if coord:
                current_shuttles[sid] = (coord["lon"], coord["lat"], occ)
        except: pass
        latest_time = row["currenttime"]
    current_index = len(all_rows)
    return n

# 5) 지도 업데이트 (밀집도 + 거절 히트맵 포함)
# ──────────────────────────────────────────────────────────────
@app.callback(
    Output("map-graph","figure"),
    Output("current-time-display","children"),
    Input("interval","n_intervals"),
    State("show-density","data"),
    State("show-reject-heatmap","data")
)
def update_map(n_intervals, show_density, show_reject):
    fig = go.Figure()

    # (0) 중심용 투명 마커
    fig.add_trace(go.Scattermapbox(
        lat=[37.29994127489433], lon=[126.83721112385808],
        mode="markers", marker=dict(size=0), hoverinfo="skip"
    ))

    # (1) 셔틀 경로
    for sid, pts in shuttle_paths.items():
        if len(pts) >= 2:
            c = assign_shuttle_color(sid)
            fig.add_trace(go.Scattermapbox(
                lon=[p["lon"] for p in pts],
                lat=[p["lat"] for p in pts],
                mode="lines",
                line=dict(width=3, color=c),
                hoverinfo="skip"
            ))

    # (2) 셔틀 위치
    for sid, (lon, lat, occ) in current_shuttles.items():
        if len(shuttle_paths.get(sid, [])) <= 1:
            marker_color = text_color = "gray"
        else:
            marker_color = text_color = assign_shuttle_color(sid)

        fig.add_trace(go.Scattermapbox(
            lon=[lon], lat=[lat],
            mode="markers+text",
            marker=dict(size=14, color=marker_color),
            text=[f"S{sid[-1]} : {occ}명"],
            textposition="top center",
            textfont=dict(color=text_color, size=12),
            customdata=[sid]
        ))

    # (3) 일반 승객 밀집도
    if show_density and all_passenger_rows:
        counts = defaultdict(int)
        for r in all_passenger_rows:
            if r.get("calltime", 0) <= latest_time:
                try:
                    nid = int(r.get("arr_node"))
                    if nid in latlon_dict:
                        counts[nid] += 1
                except:
                    pass
        if counts:
            fig.add_trace(go.Densitymapbox(
                lat=[latlon_dict[n]["lat"] for n in counts],
                lon=[latlon_dict[n]["lon"] for n in counts],
                z=list(counts.values()),
                radius=25, colorscale="YlOrRd",
                showscale=False, opacity=0.6
            ))

    # (4) 거절 히트맵 레이어
    if show_reject and all_passenger_rows:
        reject_lons = []
        reject_lats = []
        for r in all_passenger_rows:
            # 실패한 콜(request)
            if not r.get("success") and r.get("calltime", 0) <= latest_time:
                dep = r.get("dep_node-expanded","")
                try:
                    # "(lon, lat)" → ["lon"," lat"]
                    lon_str, lat_str = dep.strip("()").split(",")
                    lon = float(lon_str)
                    lat = float(lat_str)
                    reject_lons.append(lon)
                    reject_lats.append(lat)
                except Exception:
                    # 포맷이 다르면 무시
                    continue

        if reject_lons:
            fig.add_trace(go.Densitymapbox(
                lon=reject_lons,
                lat=reject_lats,
                # z는 빈도수, 포인트마다 1씩
                z=[1]*len(reject_lons),
                radius=25,
                colorscale=[
                    [0.0, "rgba(255,200,200,0.0)"],
                    [1.0, "rgba(255,0,0,0.6)"]
                ],
                showscale=False,
                opacity=0.6
            ))

        # update_map 콜백 안, (4) 거절 히트맵 뒤에 추가하세요.

       # ─── (5) 승객 이미지 레이어 (Full redraw) ──────────────────
    layers = []
    hover_traces = []
    d = 0.0015

    for r in all_passenger_rows:
        pid = r.get("passenger_id")
        call = r.get("calltime", 0)
        succ = r.get("success")
        arrv = r.get("arrivaltime")

        # 1) success가 True/False거나, 600초 초과하거나, 이미 arrivaltime 있으면 스킵
        if succ is not None or latest_time - call > 600 or arrv is not None:
            continue

        # 2) 좌표 파싱
        dep_str = r.get("dep_node_expanded","")
        try:
            lon_str, lat_str = dep_str.strip("()").split(",",1)
            lon, lat = float(lon_str), float(lat_str)
        except:
            continue

        # 3) 아이콘 경로 (1번 방식: 동적 경로 생성)
        sid = r.get("shuttleid")
        if sid:
            color = assign_shuttle_color(sid)     # 실제 셔틀 색(hex)
            clean = color.lstrip("#")
            icon = f"/assets/passenger_{clean}.png"
        else:
            icon = "/assets/passenger_gray.png"

        # 4) image layer
        layers.append({
            "sourcetype":"image",
            "source": icon,
            "coordinates":[
                [lon-d, lat+d],
                [lon+d, lat+d],
                [lon+d, lat-d],
                [lon-d, lat-d],
            ]
        })

        # 5) hover용 invisible scatter
        occ = get_psgrnum_for_shuttle(pid)
        hover_traces.append(go.Scattermapbox(
            lon=[lon], lat=[lat],
            mode="markers",
            marker=dict(size=2, opacity=0),
            customdata=[[pid, occ]],
            hovertemplate=(
                "승객 ID: %{customdata[0]}<br>"
                "탑승객 수: %{customdata[1]}명"
                "<extra></extra>"
            ),
            showlegend=False
        ))

    # 6) 레이아웃 업데이트
    fig.update_layout(
        mapbox_layers=layers,
        mapbox_style="carto-positron",
        mapbox_center=dict(lat=37.29994, lon=126.83721),
        mapbox_zoom=12,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )

    # 7) hover traces 추가
    for tr in hover_traces:
        fig.add_trace(tr)

    return fig, f"최근 업데이트: {latest_time:.0f}"

# ──────────────────────────────────────────────────────────────
# 6) KPI 업데이트 콜백
# ──────────────────────────────────────────────────────────────
@app.callback(
    [
        Output("total-shuttles",            "children"),
        Output("avg-occupancy",             "children"),
        Output("cumulative_passenger_num", "children"),
        Output("rejection-rate",            "children"),
    ],
    Input("interval","n_intervals")
)
def update_kpi(n):
    running = count_running_shuttles(current_shuttles, shuttle_paths)
    avg     = compute_avg_occupancy(current_shuttles, shuttle_paths)
    pct     = compute_rejection_rate(cur, latest_time)
    total_served = sum(
        r.get("psgrnum", 1)  # psgrnum 키가 없으면 기본 1명
        for r in all_passenger_rows
        if r.get("success") in (1, True)
           and r.get("calltime", 0) <= latest_time
    )
    cumul_num = int(total_served)
    total_children = [
        html.Span(str(running)), html.Small("대 / "),
        html.Span(str(TOTAL_SHUTTLES)), html.Small("대"),
    ]
    avg_children = [
        html.Span(f"{avg:.1f}"), html.Small("명"),
    ]
    cum_children = [
        html.Span(str(cumul_num)), html.Small("명"),
    ]
    rej_children = [
        html.Span(f"{pct:.1f}"), html.Small("%"),
    ]
    return total_children, avg_children, cum_children, rej_children

# ──────────────────────────────────────────────────────────────
# 7) 챗봇 메시지 처리 콜백
# ──────────────────────────────────────────────────────────────
@app.callback(
    Output("chat-output","children"),
    [Input("chat-send","n_clicks"),
     State("chat-input","value"),
     State("chat-output","children")],
    prevent_initial_call=True
)
def handle_chat(n_clicks, user_msg, history):
    if not user_msg or user_msg.strip()=="":
        raise PreventUpdate

    # Flask REST API로 POST 요청
    try:
        # 1. 요청 설정
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # 2. POST 요청 전송
        response = requests.post(
            "http://127.0.0.1:6789/chat-api",
            json={"request_message": user_msg},
            headers=headers,
            timeout=15  # 타임아웃 15초로 설정
        )
        
        # 3. 응답 확인
        response.raise_for_status()
        bot_msg = response.json().get("response_message", "응답 오류")
        
    except requests.exceptions.Timeout:
        bot_msg = "응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."
    except requests.exceptions.ConnectionError:
        bot_msg = "서버에 연결할 수 없습니다. Flask 서버가 실행 중인지 확인해주세요."
    except Exception as e:
        bot_msg = f"오류가 발생했습니다: {str(e)}"

    def bubble(text, who):
        """
        who == "user" → class="chat-bubble user"
        who == "bot"  → class="chat-bubble bot"
        CSS 파일에서 .chat-bubble.user / .chat-bubble.bot 규칙을 정의해 두었으므로
        인라인 스타일은 빼고, className만 붙여 줍니다.
        """
        class_name = f"chat-bubble {who}"  # "chat-bubble user" 또는 "chat-bubble bot"
        return html.Div(
            html.Span(text, className=class_name),
            # 텍스트 정렬을 위해만 style 남김
            style={"textAlign": "right" if who=="user" else "left", "marginBottom": "8px"}
        )


    return (history or []) + [bubble(user_msg,"user"), bubble(bot_msg,"bot")]

# ──────────────────────────────────────────────────────────────
# 8) 셔틀 클릭 → 상세 카드 콜백
# ──────────────────────────────────────────────────────────────
@app.callback(
    Output("shuttle-info-card","children"),
    [Input("map-graph","clickData"), Input("close-shuttle-btn","n_clicks")]
)
def display_shuttle_card(clickData, close_clicks):
    trig = callback_context.triggered_id
    if trig == "close-shuttle-btn":
        return [html.Button("", id="close-shuttle-btn", n_clicks=0,
                            style={"display":"none"})]
    if trig == "map-graph" and clickData and "points" in clickData:
        sid = clickData["points"][0].get("customdata")
        if not sid: raise PreventUpdate
        lon, lat, occ = current_shuttles.get(sid, (None,None,None))
        if lon is None: raise PreventUpdate

        curr_name = next(
            (name for (x,y),name in coord_to_name.items()
             if abs(x-lon)<1e-4 and abs(y-lat)<1e-4),
            "알 수 없음"
        )
        path = shuttle_paths.get(sid, [])
        next_name = path[1]["node_name"] if len(path)>1 else "정보 없음"
        metrics = compute_shuttle_metrics(all_rows, sid, latlon_dict)

        card = dbc.Card([
            dbc.CardHeader([
                html.Span("셔틀 상세 정보"),
                html.Button("×", id="close-shuttle-btn", n_clicks=0,
                            style={"float":"right","border":"none",
                                   "background":"transparent","fontSize":"18px",
                                   "lineHeight":"1","padding":"0 6px","cursor":"pointer"})
            ]),
            dbc.CardBody([
                html.P(f"ID: {sid}", className="card-title"),
                html.P(f"현재 위치: {curr_name}"),
                html.P(f"탑승 인원: {occ}명"),
                html.P(f"다음 위치: {next_name}"),
                html.Hr(),
                html.P(f"평균 탑승자: {metrics['average_occupancy']}명"),
                html.P(f"운행 시간 (h): {metrics['run_time_h']}"),
                #html.P(f"정체 시간 (h): {metrics['idle_time_h']}"),
                #html.P(f"누적 거리 (km): {metrics['total_distance_km']}"),
            ])
        ], color="light", outline=True)
        return [card]

    raise PreventUpdate

@app.callback(
    Output("table-passengers", "data"),
    [
        Input("filter-passenger-id", "value"),
        Input("filter-success", "value"),
        Input("filter-shuttle-id", "value"),
    ]
)
def update_passenger_table(pid, succ, sq):
    """
    all_passenger_rows에서 pid, success, shuttleid 필터를 적용한 후
    결과를 반환합니다.
    """
    return data_utils.get_passenger_records(
        rows=all_passenger_rows,
        pid=pid,
        success_vals=succ,
        shuttle_query=sq
    )


@app.callback(Output("table-vehicles","data"),
              Input("filter-shuttle-id","value"))
def update_vehicle_table(sq):
    return data_utils.get_vehicle_records(all_rows, shuttle_query=sq)

@app.callback(
    Output("wait-error-graph", "figure"),
    Input("interval", "n_intervals")
)

####################################################
def update_wait_error_graph(n):
    df = pd.DataFrame(all_passenger_rows)
    df = df[df["success"].isin([True, 1])]
    if df.empty or not {"boardingtime","calltime","expectedwaitingtime"}.issubset(df.columns):
        return go.Figure()

    df = df.dropna(subset=["boardingtime","calltime","expectedwaitingtime"])
    df["actual_wait"] = df["boardingtime"] - df["calltime"]
    df["wait_error"] = df["expectedwaitingtime"] - df["actual_wait"]

    # 히스토그램 그리기
    fig = px.histogram(
        df,
        x="wait_error",
        nbins=30,
        title="예상 대기시간 – 실제 대기시간 분포",
        labels={"wait_error": "대기시간 차이 (초)"},    # x축 레이블
        color_discrete_sequence=["blue"],            # 막대 파란색
        template="plotly_white"                      # 배경 흰색 테마
    )

    # y축 제목을 "횟수"로 변경, 여백 조정
    fig.update_layout(
        yaxis_title="횟수",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig
@app.callback(
    Output("travel-error-graph", "figure"),
    Input("interval", "n_intervals")
)
def update_travel_error_graph(n):
    # 1) DataFrame 생성
    df = pd.DataFrame(all_passenger_rows)

    df = df[df["success"].isin([True, 1])]
    # 2) 필요한 컬럼 검사
    need = {"boardingtime", "arrivaltime", "expectedarrivaltime"}
    if df.empty or not need.issubset(df.columns):
        return go.Figure()

    # 3) 실제 이동시간 계산 & 오차 계산
    df = df.dropna(subset=list(need))
    df["actual_travel"] = df["arrivaltime"] - df["boardingtime"]
    df["travel_error"]  = df["expectedarrivaltime"] - df["actual_travel"]

    # 4) 히스토그램 그리기
    fig = px.histogram(
        df,
        x="travel_error",
        nbins=30,
        title="예상 이동시간 – 실제 이동시간 분포",
        labels={"travel_error": "이동시간 차이 (초)"},
        color_discrete_sequence=["blue"],  # 파란색
        template="plotly_white"            # 흰 배경
    )
    fig.update_layout(
        yaxis_title="횟수",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ──────────────────────────────────────────────────────────────
# 3) update_rej_area_table 콜백: node_id → 한글 이름 매핑 적용
# ──────────────────────────────────────────────────────────────
@app.callback(
    Output("rej-area-table","data"),
    Input("interval","n_intervals")
)
def update_rej_area_table(n):
    from collections import Counter

    failed = [r for r in all_passenger_rows if r.get("success") is False]
    cnt = Counter()
    for r in failed:
        dep = r.get("dep_node_expanded","")
        try:
            lon, lat = map(float, dep.strip("()").split(",",1))
        except:
            continue
        nid = find_nearest_nodes(lon, lat, node_data, 1)[0]
        cnt[nid] += 1

    top10 = [(nid,c) for nid,c in cnt.items() if c>=1]
    top10.sort(key=lambda x: x[1], reverse=True)
    top10 = top10[:10]

    rows = []
    for nid, c in top10:
        # 문자열 nid → 정수로 변환 시도
        try:
            lookup_id = int(nid)
        except ValueError:
            lookup_id = nid

        # mapping.csv 의 int id와 매칭
        name = node_name_map.get(lookup_id, nid)
        rows.append({"region": name, "count": c})

    return rows


# ──────────────────────────────────────────────────────────────
# 10. 실행
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False,port=7777)
