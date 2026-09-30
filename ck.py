import os
import uuid
import warnings
import importlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

warnings.filterwarnings("ignore")

try:
    import xarray as xr
    HAS_XR = True
except ImportError:
    HAS_XR = False

try:
    import cartopy
    from cartopy.io.shapereader import Reader
    from shapely.ops import unary_union
    from shapely import contains_xy
    from shapely.prepared import prep as _shapely_prep
    HAS_COASTLINE_MASK = True
except ImportError:
    HAS_COASTLINE_MASK = False

try:
    from scipy.spatial import cKDTree
    HAS_KD_TREE = True
except ImportError:
    HAS_KD_TREE = False

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    layout="wide",
    page_title="海洋塑膠垃圾溯源分析系統",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTS & STRICT PATHS
# ============================================================

METERS_PER_DEG_LAT = 111320.0

# ------------------------------------------------------------
# 物理合理性包絡檢驗 (Plausibility Envelope QC) 上限常數
# ------------------------------------------------------------
# 海流分量物理包絡上限：3.0 m/s
#   依據：黑潮主流表層流速 1.5~2.0 m/s；中尺度渦旋（mesoscale eddy）
#         疊加後極限約 2.5 m/s。取 3.0 m/s 保留極值餘裕，
#         同時過濾模式邊界/地形格點的數值異常（spike）。
CURRENT_SPEED_CAP_MPS = 3.0

# 風場分量物理包絡上限：35.0 m/s
#   依據：WMO 蒲福風級 12 級（颱風門檻 32.7 m/s）。
#         冬季台灣海峽東北季風 10m 風速常達 8~15 m/s，
#         強鋒面/颱風可逾 30 m/s；取 35.0 m/s 完整包絡冬季極端事件。
WIND_SPEED_CAP_MPS = 35.0

# 向後相容別名：舊版以 SPEED_CAP_MPS 代表「合成總速度」上限，
# 語意已由雙層 QC 架構取代（見 rk4_step_2d.get_uv）。
SPEED_CAP_MPS = CURRENT_SPEED_CAP_MPS

AUTO_OFFSHORE_START_KM = 100.0
AUTO_OFFSHORE_STEP_M = 500.0

# 沿岸關聯半徑（政策法理依據）：
# 反向回溯時「觸岸粒子」已由 STATUS_BEACHED 精確捕獲；
# 此半徑僅用於判定「回溯期滿仍漂浮於近岸領海滯留帶」之粒子。
# 預設 22.2 km = 12 浬法定領海責任水域，可由側邊欄滑桿於 10~50 km 調整。
LOCAL_SOURCE_RADIUS_KM_DEFAULT = 22.2
LOCAL_SOURCE_RADIUS_KM_MIN = 10.0
LOCAL_SOURCE_RADIUS_KM_MAX = 50.0
# 保留模組層級常數以維持既有引用相容（實際判定值由側邊欄傳入）。
LOCAL_SOURCE_RADIUS_KM = LOCAL_SOURCE_RADIUS_KM_DEFAULT

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."

# ------------------------------------------------------------
# 資料來源解析 (Data Source Resolution)
# ------------------------------------------------------------
# 本系統僅使用「本機精簡資料」(slim_data/)：
#   由 prepare_local_data.py 從原始 NODASS 資料庫萃取，
#   只保留專案實際使用的欄位（POM us/vs、WRF u10/v10），
#   約 1.15 GB，可完整存放於本機，無需外接硬碟。
#   檔案：slim_data/pom_currents_slim.nc、slim_data/wrf_wind_slim.nc
#
# 設計取捨：不再支援原始資料庫佈局（POM/ + WRF/，約 424 GB）。
#   原因：原始佈局需合併 31 個 POM 檔並解析 124 個 GRIB2，
#   載入耗時數十秒至數分鐘，且依賴外接硬碟路徑，部署脆弱。
#   精簡資料已由 verify_local_data.py 逐點驗證與原始資料一致
#   （最大絕對誤差 <= 1e-5），故直接以精簡資料為唯一來源。
# ------------------------------------------------------------
SLIM_DIR = os.path.join(BASE_DIR, "slim_data")
SLIM_POM_FILE = os.path.join(SLIM_DIR, "pom_currents_slim.nc")
SLIM_WRF_FILE = os.path.join(SLIM_DIR, "wrf_wind_slim.nc")

STATUS_ACTIVE = 0
STATUS_BEACHED = 1
STATUS_OUT_OF_BOUNDS = 2
STATUS_TIME_EXCEEDED = 3
STATUS_OOB = STATUS_OUT_OF_BOUNDS

# 回溯物理狀態標記
STATUS_LABELS = {
    STATUS_ACTIVE: "外海漂流中 (Offshore Active)",
    STATUS_BEACHED: "沿岸陸源釋放 (Terrestrial Origin)",
    STATUS_OUT_OF_BOUNDS: "超出模擬邊界 (Out of Bounds)",
    STATUS_TIME_EXCEEDED: "資料時間軸耗盡 (Time Exceeded)",
}

# ============================================================
# VISUAL STYLE
# ============================================================

HOTSPOT_FILL_COLOR = [255, 140, 0, 90]
HOTSPOT_EDGE_COLOR = [255, 255, 255, 180]
HOTSPOT_RADIUS = 300

CURRENT_LOCAL_COLOR = [52, 152, 219, 255]       
CURRENT_EXTERNAL_COLOR = [231, 76, 60, 255]     
PATH_LOCAL_COLOR = [52, 152, 219, 220]
PATH_EXTERNAL_COLOR = [231, 76, 60, 220]

CURRENT_POINT_RADIUS = 80
PATH_WIDTH_MIN_PIXELS = 3

# ============================================================
# HOTSPOTS
# ============================================================

TAIWAN_BEACHING_SITES = {
    "北部 - 基隆／野柳": ((121.60, 25.43), (121.90, 24.98)),
    "北部 - 淡水／八里": ((121.35, 25.15), (121.45, 25.25)),
    "東部 - 宜蘭": ((121.80, 24.70), (121.90, 24.85)),
    "東部 - 花蓮": ((121.55, 24.17), (121.75, 23.67)),
    "東部 - 臺東": ((121.28, 22.95), (121.45, 22.70)),
    "南部 - 高雄港": ((120.25, 22.60), (120.30, 22.65)),
    "南部 - 墾丁": ((120.70, 21.90), (120.85, 22.05)),
    "西部 - 臺南": ((120.10, 23.00), (120.20, 23.15)),
    "西部 - 臺中": ((120.45, 24.20), (120.60, 24.40)),
    "西部 - 新竹": ((120.70, 25.00), (120.90, 24.70)),
    "離島 - 澎湖": ((119.50, 23.50), (119.70, 23.70)),
    "離島 - 小琉球": ((120.35, 22.32), (120.39, 22.36)),
}

# ------------------------------------------------------------
# 熱點釋放座標 (Release Coordinates)
# ------------------------------------------------------------
# 注意：不可使用 bounding-box 的幾何中心作為釋放點。
# 台灣多數監測熱點為「沿海線狀段」，其外接矩形中心常落在陸地上
# （例如高雄港、臺南、臺中、澎湖、小琉球、墾丁），
# 導致粒子初始即位於陸地、速度場為零、位移恆為 0。
#
# 以下座標為各熱點「代表性近岸海域點」，位於該段海岸線外側的
# 可航行水域，確保粒子釋放後能受海流驅動。
# 座標來源：以 Cartopy Natural Earth 10m 陸地多邊形為基準，
#           自各熱點海岸線外推至距陸地約 2.5 km 之海域點
#           （約 1–1.5 個網格，網格解析度 0.02° ≈ 2.2 km）。
#           澎湖、小琉球原點已足夠近岸（2.2–2.4 km），維持不變。
# ------------------------------------------------------------
HOTSPOT_RELEASE_POINTS = {
    "北部 - 基隆／野柳": (121.9088, 25.1305),   # 基隆外海（東北角）
    "北部 - 淡水／八里": (121.3114, 25.1504),   # 淡水河口外側
    "東部 - 宜蘭": (121.9510, 24.8137),         # 蘭陽平原外海
    "東部 - 花蓮": (121.6296, 23.9533),         # 花蓮外海
    "東部 - 臺東": (121.2713, 22.8816),         # 臺東外海
    "南部 - 高雄港": (120.2453, 22.6096),       # 高雄港外側海域
    "南部 - 墾丁": (120.7243, 21.9122),         # 墾丁南灣外海
    "西部 - 臺南": (120.0272, 23.0286),         # 臺南安平外海
    "西部 - 臺中": (120.4902, 24.3013),         # 臺中港外海
    "西部 - 新竹": (120.8805, 24.8423),         # 新竹外海
    "離島 - 澎湖": (119.5200, 23.5200),         # 澎湖群島西南海域
    "離島 - 小琉球": (120.3400, 22.3000),       # 小琉球西南海域
}

# 向後相容：HOTSPOT_CENTERS 保留名稱，但改指向已校正的釋放座標。
HOTSPOT_CENTERS = dict(HOTSPOT_RELEASE_POINTS)

# ============================================================
# PLOT EXPORT
# ============================================================

PLOTLY_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "plastic_source_chart",
        "width": 1400,
        "height": 800,
        "scale": 2,
    },
}

# ============================================================
# TITLE
# ============================================================

st.title("海洋塑膠垃圾溯源分析系統")

# ============================================================
# HELPERS
# ============================================================

def _extract_surface_2d(var_data):
    if var_data.ndim == 4:
        return var_data[:, 0, :, :]
    elif var_data.ndim == 3 and not any(k in str(var_data.dims[0]).lower() for k in ["time", "step"]):
        return var_data[0, :, :]
    return var_data


def _as_time_space_array(var_data):
    """Return a variable as (time, y, x), retaining curvilinear coordinates."""
    values = np.asarray(var_data.values if hasattr(var_data, "values") else var_data)
    if values.ndim < 2:
        raise ValueError("風流變數至少需要二維空間資料")
    if values.ndim > 2:
        return values.reshape((-1,) + values.shape[-2:])
    return values[np.newaxis, ...]


def _normalise_datetime_coord(ds, time_name):
    if not time_name or time_name not in ds:
        return ds, time_name
    try:
        ds[time_name] = xr.decode_cf(ds)[time_name]
    except Exception:
        pass
    return ds, time_name

def meters_to_deg_lon(m, lat):
    return m / (METERS_PER_DEG_LAT * np.cos(np.deg2rad(lat)) + 1e-12)

def meters_to_deg_lat(m):
    return m / METERS_PER_DEG_LAT

def haversine(lon1, lat1, lon2, lat2):
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)

    R = 6371000.0
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def _pick_first_match(cands, available):
    lower = {k.lower(): k for k in available}
    for n in cands:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def _inside_bbox(lon, lat, bbox):
    return bbox[0] <= lon <= bbox[1] and bbox[2] <= lat <= bbox[3]


def _pick_first_from_candidates(candidates, available):
    normalized = {str(k).lower(): str(k) for k in available}
    for cand in candidates:
        key = str(cand).lower()
        if key in normalized:
            return normalized[key]
    return None


def _safe_time_to_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).to_pydatetime().replace(tzinfo=timezone.utc)
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            value = value
    if isinstance(value, str):
        try:
            parsed = pd.Timestamp(value)
            if parsed.tzinfo is None:
                parsed = parsed.tz_localize("UTC")
            else:
                parsed = parsed.tz_convert("UTC")
            return parsed.to_pydatetime()
        except Exception:
            return None
    try:
        timestamp = pd.Timestamp(value)
        if pd.notna(timestamp):
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            return timestamp.to_pydatetime()
    except Exception:
        return None
    return None


def _infer_dataset_reference_time(ds, time_name, preferred_time=None):
    if preferred_time is not None:
        parsed = _safe_time_to_datetime(preferred_time)
        if parsed is not None:
            return parsed

    if time_name and time_name in ds.coords:
        try:
            max_val = ds[time_name].max().values
            parsed = _safe_time_to_datetime(max_val)
            if parsed is not None:
                return parsed
        except Exception:
            pass

    if time_name and time_name in ds.variables:
        try:
            max_val = ds[time_name].values
            if np.size(max_val) > 0:
                parsed = _safe_time_to_datetime(max_val.flat[-1])
                if parsed is not None:
                    return parsed
        except Exception:
            pass

    return datetime.now(timezone.utc)


def _infer_step_hours(coord):
    vals = np.asarray(coord.values)
    if len(vals) < 2:
        return 3.0

    if np.issubdtype(vals.dtype, np.datetime64):
        delta_ns = np.median(np.diff(vals).astype("timedelta64[ns]").astype(np.int64))
        return abs(float(delta_ns)) / 3.6e12
    if np.issubdtype(vals.dtype, np.timedelta64):
        delta_ns = np.median(np.diff(vals).astype("timedelta64[ns]").astype(np.int64))
        return abs(float(delta_ns)) / 3.6e12

    step = float(np.median(np.diff(vals)))
    units = str(coord.attrs.get("units", "")).lower()

    if "second" in units:
        return abs(step) / 3600.0
    if "minute" in units:
        return abs(step) / 60.0
    if "hour" in units:
        return abs(step)
    if "day" in units:
        return abs(step) * 24.0

    if abs(step) > 0:
        return abs(step)
    return 3.0

def source_kind_to_color(kind):
    if "本地" in str(kind) or kind == "Local Source":
        return CURRENT_LOCAL_COLOR
    return CURRENT_EXTERNAL_COLOR

def source_kind_to_path_color(kind):
    if "本地" in str(kind) or kind == "Local Source":
        return PATH_LOCAL_COLOR
    return PATH_EXTERNAL_COLOR

def render_map_legend():
    st.markdown(
        """
        <div style="
            background: rgba(20,20,20,0.72);
            padding: 10px 14px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.12);
            margin-bottom: 12px;
            color: white;
            font-size: 13px;
            line-height: 1.6;
        ">
            <b>圖層分類說明</b><br>
            <span style="font-size:16px; color: rgb(255,140,0);">●</span> 監測錨點熱點 &nbsp;|&nbsp; 
            <span style="font-size:16px; color: rgb(52,152,219);">●</span> 本地源軌跡 (觸岸 或 &le; 領海半徑) &nbsp;|&nbsp; 
            <span style="font-size:16px; color: rgb(231,76,60);">●</span> 境外源軌跡 (> 領海半徑)
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# LAND MASK
# ============================================================

def is_land_vectorized(lon_arr, lat_arr, prov):
    """判定座標是否落在陸地上。

    優先使用「高精度幾何多邊形」直接查詢（Natural Earth 10m，精度約 10 m），
    而非查表 POM 網格遮罩（0.02° ≈ 2.2 km）。

    為何必須如此：
      POM 網格遮罩的解析度僅 2.2 km，台灣北海岸（基隆港、野柳岬、淡水河口
      沙洲）與澎湖群島等寬度小於 2.2 km 的地形會被「抹平」成水域。
      粒子實際已走進陸地，但遮罩判定為水，因此不觸發觸岸停止，
      造成「碰到陸地但沒停」的錯誤軌跡。

    幾何查詢不可用時，才回退至 POM 網格遮罩（最近鄰查表）。
    """
    geom = prov.get("curr", {}).get("land_geometry")
    if geom is not None:
        lon_a = np.asarray(lon_arr, dtype=np.float64)
        lat_a = np.asarray(lat_arr, dtype=np.float64)
        return np.asarray(contains_xy(geom, lon_a, lat_a), dtype=bool)

    lat_ax = prov["curr"]["lat"]
    lon_ax = prov["curr"]["lon"]
    mask = prov["curr"]["landmask"]

    if np.asarray(lat_ax).ndim == 2 or np.asarray(lon_ax).ndim == 2:
        lat2 = np.asarray(lat_ax)
        lon2 = np.asarray(lon_ax)
        distance = (lon2[None, ...] - np.asarray(lon_arr).reshape(-1, 1, 1)) ** 2
        distance += (lat2[None, ...] - np.asarray(lat_arr).reshape(-1, 1, 1)) ** 2
        nearest = np.nanargmin(distance.reshape(len(np.asarray(lon_arr)), -1), axis=1)
        return np.asarray(mask).reshape(-1)[nearest]

    idx_x = np.clip(np.round((lon_arr - lon_ax[0]) / (lon_ax[1] - lon_ax[0] + 1e-12)).astype(int), 0, len(lon_ax) - 1)
    idx_y = np.clip(np.round((lat_arr - lat_ax[0]) / (lat_ax[1] - lat_ax[0] + 1e-12)).astype(int), 0, len(lat_ax) - 1)
    return mask[idx_y, idx_x]


def build_public_coastline_mask(lat, lon, return_geometry=False):
    """Build a reproducible land mask from Cartopy's local Natural Earth cache.

    return_geometry=True 時額外回傳 (mask, geometry)，讓呼叫端可保存高精度
    幾何物件，供 is_land_vectorized 直接查詢（不受 POM 網格解析度限制）。
    """
    if not HAS_COASTLINE_MASK:
        raise RuntimeError("当前 Python 环境没有 cartopy/shapely，无法建立公开海岸线遮罩")

    # 依序尋找海岸線資料：
    #   1) 專案內打包的 cartopy_data/（部署環境用，確保雲端可用）
    #   2) Cartopy 預設資料目錄（本機開發環境）
    candidate_paths = [
        os.path.join(
            BASE_DIR,
            "cartopy_data",
            "shapefiles",
            "natural_earth",
            "physical",
            "ne_10m_land.shp",
        ),
        os.path.join(
            str(cartopy.config["data_dir"]),
            "shapefiles",
            "natural_earth",
            "physical",
            "ne_10m_land.shp",
        ),
    ]
    land_path = next((p for p in candidate_paths if os.path.isfile(p)), None)
    if land_path is None:
        raise FileNotFoundError(
            "找不到 Natural Earth 海岸线资料（已检查专案内 cartopy_data/ 与 "
            f"Cartopy 预设目录 {cartopy.config['data_dir']}）；"
            "为避免不可追溯下载，程序不会自动联网取得资料。"
        )

    geometry = unary_union(list(Reader(land_path).geometries()))
    lat_values = np.asarray(lat, dtype=np.float64)
    lon_values = np.asarray(lon, dtype=np.float64)
    if lat_values.ndim == 1 and lon_values.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    elif lat_values.shape == lon_values.shape:
        lat_grid, lon_grid = lat_values, lon_values
    else:
        raise ValueError("POM 经纬度网格形状不一致，无法建立海岸线遮罩")
    mask = np.asarray(contains_xy(geometry, lon_grid, lat_grid), dtype=bool)
    if return_geometry:
        return mask, geometry
    return mask

# ============================================================
# DATA PROVIDERS (SLIM DATA LOADER)
# ============================================================

def _build_providers_from_slim(need_wind, open_nc, landmask_from_velocity):
    """從本機精簡資料集 (slim_data/) 建立速度場提供者。

    精簡資料由 prepare_local_data.py 產生，只保留專案實際使用的欄位：
      - pom_currents_slim.nc : us, vs, lat, lon, time
      - wrf_wind_slim.nc     : u10, v10, latitude, longitude, time

    設計要點：
      - 直接讀取單一 NetCDF，載入時間為秒級（無需合併 31 個 POM 檔、
        亦無需解析 124 個 GRIB2）。
      - 海陸遮罩以 Cartopy Natural Earth 10m 幾何多邊形為最高準則，
        確保陸地判定不受 POM 網格解析度（0.02° ≈ 2.2 km）限制。
    """
    # ---------------- POM 海流 ----------------
    if not os.path.isfile(SLIM_POM_FILE):
        st.error(f"找不到精簡 POM 檔案：`{SLIM_POM_FILE}`。請執行 `python prepare_local_data.py` 重新產生。")
        st.stop()

    try:
        ds_ocean = open_nc(SLIM_POM_FILE)
        names = list(ds_ocean.variables)
        lat_name = _pick_first_from_candidates(["lat", "latitude", "lat_rho", "y"], names)
        lon_name = _pick_first_from_candidates(["lon", "longitude", "lon_rho", "x"], names)
        u_name = _pick_first_from_candidates(["us", "u", "water_u", "u_eastward"], names)
        v_name = _pick_first_from_candidates(["vs", "v", "water_v", "v_northward"], names)
        time_name = _pick_first_from_candidates(["time", "ocean_time", "valid_time"], names)
        if not all([lat_name, lon_name, u_name, v_name, time_name]):
            raise ValueError(f"精簡 POM 缺少必要欄位；目前欄位={names}")

        ds_ocean = ds_ocean.sortby(time_name)
        time_values = pd.to_datetime(np.asarray(ds_ocean[time_name].values))
        keep = ~pd.Index(time_values).duplicated(keep="first")
        ds_ocean = ds_ocean.isel({time_name: np.flatnonzero(keep)})

        ocean_dt_hours = _infer_step_hours(ds_ocean[time_name])
        reference_time = _safe_time_to_datetime(ds_ocean[time_name].max().values)

        lat = np.asarray(ds_ocean[lat_name].values, dtype=np.float32)
        lon = np.asarray(ds_ocean[lon_name].values, dtype=np.float32)
        u = _as_time_space_array(ds_ocean[u_name]).astype(np.float32)
        v = _as_time_space_array(ds_ocean[v_name]).astype(np.float32)
        if u.shape != v.shape:
            raise ValueError(f"精簡 POM U/V 空間形狀不一致：{u.shape} / {v.shape}")

        if lat.ndim == 1 and lon.ndim == 1:
            if lat[0] > lat[-1]:
                lat, u, v = lat[::-1], np.flip(u, 1), np.flip(v, 1)
            if lon[0] > lon[-1]:
                lon, u, v = lon[::-1], np.flip(u, 2), np.flip(v, 2)

        # 海陸遮罩：與原始路徑一致，優先使用 Cartopy 幾何多邊形。
        # 同時保存原始幾何物件供 is_land_vectorized 高精度查詢。
        landmask = None
        land_geometry = None
        landmask_source = "unknown"
        landmask_confidence = "low"
        geometric_exc = None
        try:
            landmask, land_geometry = build_public_coastline_mask(lat, lon, return_geometry=True)
            landmask_source = "Cartopy Natural Earth 10m local land polygons (geometric priority)"
            landmask_confidence = "high"
        except (FileNotFoundError, ImportError, RuntimeError, ValueError) as geo_exc:
            geometric_exc = geo_exc
            landmask = landmask_from_velocity(u, v)
            landmask_source = f"POM velocity finite-value fallback (low confidence: {geometric_exc})"
            landmask_confidence = "low"

        if landmask.shape != u.shape[1:]:
            raise ValueError(
                f"海陆遮罩形状 {landmask.shape} 与流速场空间形状 {u.shape[1:]} 不一致，"
                "无法安全进行陆地判定；请检查精簡 POM 网格。"
            )

        curr = {
            "lat": lat, "lon": lon,
            "u": np.nan_to_num(u), "v": np.nan_to_num(v),
            "landmask": landmask,
            "landmask_source": landmask_source,
            "landmask_confidence": landmask_confidence,
            "land_geometry": land_geometry,
            "bbox": (float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))),
            "n_times": u.shape[0], "dt_hours": ocean_dt_hours, "reference_time_utc": reference_time,
            "time_interval_hours": ocean_dt_hours,
            "missing_fraction": float(np.mean(~np.isfinite(u) | ~np.isfinite(v))),
            "max_speed_mps": float(np.nanmax(np.hypot(u, v))),
            "time_start_utc": _safe_time_to_datetime(ds_ocean[time_name].min().values),
            "time_end_utc": _safe_time_to_datetime(ds_ocean[time_name].max().values),
            "units": {
                "u": str(ds_ocean[u_name].attrs.get("units", "m s-1")),
                "v": str(ds_ocean[v_name].attrs.get("units", "m s-1")),
            },
        }
    except Exception as exc:
        st.error(f"精簡 POM 海流讀取失敗：`{SLIM_POM_FILE}`。詳細錯誤：{exc}")
        st.stop()

    # ---------------- WRF 風場 ----------------
    wind_data = None
    wind_path = None
    if need_wind:
        if not os.path.isfile(SLIM_WRF_FILE):
            st.error(f"找不到精簡 WRF 檔案：`{SLIM_WRF_FILE}`。請執行 `python prepare_local_data.py` 重新產生。")
            st.stop()
        try:
            ds_wind = open_nc(SLIM_WRF_FILE)
            wnames = list(ds_wind.variables)
            wu_name = _pick_first_from_candidates(["u10", "10u", "u"], wnames)
            wv_name = _pick_first_from_candidates(["v10", "10v", "v"], wnames)
            wlat_name = _pick_first_from_candidates(["latitude", "lat"], wnames)
            wlon_name = _pick_first_from_candidates(["longitude", "lon"], wnames)
            wtime_name = _pick_first_from_candidates(["time", "valid_time"], wnames)
            if not all([wu_name, wv_name, wlat_name, wlon_name, wtime_name]):
                raise ValueError(f"精簡 WRF 缺少必要欄位；目前欄位={wnames}")

            ds_wind = ds_wind.sortby(wtime_name)
            wind_times = pd.to_datetime(np.asarray(ds_wind[wtime_name].values))
            wind_u = _as_time_space_array(ds_wind[wu_name]).astype(np.float32)
            wind_v = _as_time_space_array(ds_wind[wv_name]).astype(np.float32)
            if wind_u.shape != wind_v.shape:
                raise ValueError(f"精簡 WRF U/V 形狀不一致：{wind_u.shape} / {wind_v.shape}")

            wind_lat = np.asarray(ds_wind[wlat_name].values, dtype=np.float32)
            wind_lon = np.asarray(ds_wind[wlon_name].values, dtype=np.float32)

            dt_hours = (
                float(np.median(np.diff(wind_times.astype("int64"))) / 3.6e12)
                if len(wind_times) > 1 else 1.0
            )
            wind_gaps = np.diff(wind_times.astype("int64")) / 3.6e12
            if len(wind_gaps) and float(np.max(wind_gaps)) > max(dt_hours * 1.5, dt_hours + 1e-6):
                st.error(
                    "精簡 WRF 分析時次不連續："
                    f"中位間隔 {dt_hours:.1f} 小時，最大間隔 {float(np.max(wind_gaps)):.1f} 小時。"
                    "請重新執行 `python prepare_local_data.py` 產生完整資料。"
                )
                st.stop()

            spatial_index = build_curvilinear_index(wind_lat, wind_lon)
            wind_data = {
                "lat": wind_lat, "lon": wind_lon,
                "u": np.nan_to_num(wind_u), "v": np.nan_to_num(wind_v),
                "n_times": len(wind_times),
                "dt_hours": max(dt_hours, 1e-6),
                "time_start_utc": _safe_time_to_datetime(wind_times[0]),
                "time_end_utc": _safe_time_to_datetime(wind_times[-1]),
                "units": {"u": "m s-1", "v": "m s-1"},
                "time_interval_hours": dt_hours,
                "selection_rule": "slim_data/wrf_wind_slim.nc (WRF SFC _0000 analysis, 10m u10/v10)",
                "missing_fraction": float(np.mean(~np.isfinite(wind_u) | ~np.isfinite(wind_v))),
                "max_speed_mps": float(np.nanmax(np.hypot(wind_u, wind_v))),
                "spatial_index": spatial_index,
                "interpolation_method": (
                    "curvilinear cKDTree nearest-neighbor"
                    if spatial_index is not None
                    else "curvilinear axis approximation"
                ),
            }
            wind_path = SLIM_WRF_FILE
        except Exception as exc:
            st.error(f"精簡 WRF 風場讀取失敗：`{SLIM_WRF_FILE}`。詳細錯誤：{exc}")
            st.stop()

    common_end = curr["time_end_utc"]
    if wind_data is not None and wind_data.get("time_end_utc") is not None:
        common_end = min(common_end, wind_data["time_end_utc"])

    return {
        "curr": curr,
        "wind": wind_data,
        "ocean_file": os.path.basename(SLIM_POM_FILE),
        "wind_file": os.path.basename(wind_path) if wind_path else None,
        "reference_time_utc": common_end,
        "data_mode": "slim",
    }


@st.cache_resource(show_spinner=True)
def build_velocity_providers(need_wind=False):
    # 注意：此函式「不」接受 required_hours 參數。
    # 原因：@st.cache_resource 以「函式簽名 + 引數值」作為快取鍵；
    # 若把 required_hours（= 回溯天數 * 24）納入簽名，使用者每次調整
    # 回溯天數都會產生新的快取鍵，導致 745 個 POM 檔 + 124 個 WRF GRIB2
    # 全量重新載入（數十秒至數分鐘）。資料載入與回溯天數無關，
    # 因此移除該參數，讓快取鍵固定為 need_wind，切換天數時直接命中快取。
    if not HAS_XR:
        st.error("系統缺少 xarray，無法讀取 NetCDF / GRIB2 資料。請在目前執行 Streamlit 的 Python 環境安裝 xarray。")
        st.stop()

    # ------------------------------------------------------------
    # 依賴引擎前置健康檢查 (Dependency Pre-flight Check)
    # 在深入變數解析層之前，先確認環境具備至少一種 NetCDF 解析引擎，
    # 避免程式執行到一半才拋出紅字 Exception。
    # ------------------------------------------------------------
    def _detect_netcdf_engines():
        available = []
        for mod_name, engine_name in (
            ("netCDF4", "netcdf4"),
            ("scipy", "scipy"),
            ("h5netcdf", "h5netcdf"),
        ):
            try:
                importlib.import_module(mod_name)
                available.append(engine_name)
            except ImportError:
                continue
        return available

    available_engines = _detect_netcdf_engines()
    if not available_engines:
        st.error(
            "系統缺少 NetCDF 解析引擎，請在終端機執行 `pip install scipy netCDF4` 安裝後重啟。"
        )
        st.stop()

    def _landmask_from_velocity(u_values, v_values):
        u_arr = np.asarray(u_values, dtype=np.float32)
        v_arr = np.asarray(v_values, dtype=np.float32)
        valid = np.isfinite(u_arr) & np.isfinite(v_arr)
        valid &= np.abs(u_arr) < 1e20
        valid &= np.abs(v_arr) < 1e20
        # Zero current is valid water; only non-finite/fill-value cells are land.
        return ~valid[0] if valid.ndim == 3 else ~valid

    def _open_nc(path):
        """自適應 NetCDF 開啟：依序嘗試可用引擎，全部失敗才拋出清晰錯誤。

        嘗試順序：
          1. 預設模式（讓 xarray 自動選擇後端）
          2. engine="netcdf4"
          3. engine="scipy"
          4. engine="h5netcdf"
        僅在「引擎未安裝」時跳過該引擎；若引擎存在但檔案本身損毀，
        則記錄該錯誤並繼續嘗試下一個引擎。
        """
        # 依可用性排序候選引擎，避免嘗試未安裝的後端。
        candidate_engines = [None]  # None = xarray 預設自動選擇
        for engine_name in ("netcdf4", "scipy", "h5netcdf"):
            if engine_name in available_engines:
                candidate_engines.append(engine_name)

        errors = []
        for engine in candidate_engines:
            try:
                if engine is None:
                    return xr.open_dataset(path, decode_times=True)
                return xr.open_dataset(path, engine=engine, decode_times=True)
            except (ImportError, ModuleNotFoundError) as exc:
                # 引擎模組缺失：記錄後嘗試下一個。
                errors.append(f"{engine or 'default'}: 引擎未安裝 ({exc})")
                continue
            except (FileNotFoundError, OSError, ValueError) as exc:
                # 檔案或格式問題：記錄後嘗試下一個引擎。
                errors.append(f"{engine or 'default'}: {exc}")
                continue

        raise RuntimeError(
            f"無法以任何可用引擎開啟 NetCDF 檔案：`{path}`。"
            f"已嘗試引擎={candidate_engines}；"
            f"錯誤明細：{'; '.join(errors)}。"
            "請確認檔案完整性，或安裝 netCDF4 / scipy / h5netcdf 其中之一。"
        )

    # ============================================================
    # 精簡資料載入路徑 (Slim Data Path)
    # ============================================================
    # 本系統唯一資料來源：直接讀取已萃取、已壓縮的單一 NetCDF，
    # 跳過「合併 31 個 POM 檔 + 解析 124 個 GRIB2」的昂貴流程。
    # 精簡資料由 prepare_local_data.py 產生，並經 verify_local_data.py
    # 逐點驗證與原始資料一致（最大絕對誤差 <= 1e-5）。
    # ============================================================
    return _build_providers_from_slim(need_wind, _open_nc, _landmask_from_velocity)

# ============================================================
# INTERPOLATION / ENGINE
# ============================================================

def build_curvilinear_index(lat_ax, lon_ax):
    if not HAS_KD_TREE:
        return None
    lat_values = np.asarray(lat_ax, dtype=np.float64)
    lon_values = np.asarray(lon_ax, dtype=np.float64)
    if lat_values.ndim != 2 or lon_values.ndim != 2 or lat_values.shape != lon_values.shape:
        return None
    scale = np.cos(np.deg2rad(float(np.nanmean(lat_values))))
    points = np.column_stack(
        (lon_values.ravel() * scale, lat_values.ravel())
    )
    return cKDTree(points), lat_values.shape, scale


def interpolate_2d_vectorized(x, y, grid, lat_ax, lon_ax, spatial_index=None):
    lat_ax = np.asarray(lat_ax)
    lon_ax = np.asarray(lon_ax)
    # WRF/POM may expose a curvilinear (y, x) latitude/longitude grid.
    # WRF 资料网格很大，不能对每个粒子扫描整张二维网格；
    # 使用首列/首行作为近似轴，直接定位最近网格单元，计算量由 O(N*网格)
    # 降为 O(N)，同时保留二维场值的实际网格索引。
    if lat_ax.ndim == 2 or lon_ax.ndim == 2:
        lat2 = lat_ax if lat_ax.ndim == 2 else np.broadcast_to(lat_ax[:, None], lon_ax.shape)
        lon2 = lon_ax if lon_ax.ndim == 2 else np.broadcast_to(lon_ax[None, :], lat_ax.shape)
        if spatial_index is not None:
            tree, grid_shape, scale = spatial_index
            query_points = np.column_stack(
                (np.asarray(x) * scale, np.asarray(y))
            )
            _, flat_indices = tree.query(query_points)
            # cKDTree 回傳的 flat_indices 是相對於「完全攤平」座標陣列的一維索引
            # （範圍 0 ~ grid.size-1）。必須對攤平後的一維陣列取值，
            # 否則 NumPy 會把單一索引陣列作用在 axis 0（大小僅 grid_shape[0]），
            # 造成 IndexError: index ... is out of bounds for axis 0。
            grid_flat = np.asarray(grid).reshape(-1)
            # 邊界防禦：避免浮點誤差或外插導致索引越界。
            flat_indices = np.clip(flat_indices, 0, grid_flat.size - 1)
            return grid_flat[flat_indices]
        lat_axis = lat2[:, 0]
        lon_axis = lon2[0, :]
        if lat_axis[0] > lat_axis[-1]:
            lat_axis = lat_axis[::-1]
            lat_flip = True
        else:
            lat_flip = False
        if lon_axis[0] > lon_axis[-1]:
            lon_axis = lon_axis[::-1]
            lon_flip = True
        else:
            lon_flip = False
        j = np.searchsorted(lat_axis, np.asarray(y), side="left")
        i = np.searchsorted(lon_axis, np.asarray(x), side="left")
        j = np.clip(j, 0, len(lat_axis) - 1)
        i = np.clip(i, 0, len(lon_axis) - 1)
        if lat_flip:
            j = len(lat_axis) - 1 - j
        if lon_flip:
            i = len(lon_axis) - 1 - i
        return np.asarray(grid)[j, i]

    idx_x = (x - lon_ax[0]) / (lon_ax[1] - lon_ax[0] + 1e-12)
    idx_y = (y - lat_ax[0]) / (lat_ax[1] - lat_ax[0] + 1e-12)

    i0 = np.clip(np.floor(idx_x).astype(int), 0, len(lon_ax) - 2)
    j0 = np.clip(np.floor(idx_y).astype(int), 0, len(lat_ax) - 2)
    i1 = i0 + 1
    j1 = j0 + 1

    wx = idx_x - i0
    wy = idx_y - j0

    v00 = grid[j0, i0]
    v01 = grid[j0, i1]
    v10 = grid[j1, i0]
    v11 = grid[j1, i1]

    return (v00 * (1 - wx) * (1 - wy) + v01 * wx * (1 - wy) + v10 * (1 - wx) * wy + v11 * wx * wy)

def get_velocity_interpolated(lon_arr, lat_arr, field, hours_elapsed, time_offset_hours=0.0):
    dt_hours = field.get("dt_hours", 3.0)
    n_times = field["n_times"]

    # 资料时间轴按旧到新排列；逆向追溯从「使用者选定的起算时刻」开始，
    # 因此 elapsed=0 必须索引该起算时刻对应的帧，随后向较早时间移动。
    #
    # time_offset_hours = 起算时刻距「资料最后一帧」的小时数：
    #   - 起算时刻 = 资料末端 → offset = 0（旧行为）
    #   - 起算时刻 = 资料末端前 5 天 → offset = 120
    # 若忽略此偏移，无论使用者选哪一天，速度场都从最后一帧取起，
    # 导致不同日期的轨迹/位移完全相同（时间对模拟毫无影响）。
    #
    # ============================================================
    # 时间越界防护（Time Coverage Guard）
    # ============================================================
    # 资料覆盖的总时数 = (n_times - 1) * dt_hours。
    # 若 offset + elapsed 超过此跨度，代表粒子已回溯到「资料起点之前」。
    # 此时「绝不可」把 t_exact 夹到 0.0 继续积分——那会让速度场冻结在
    # 资料第一帧，粒子在静止流场中产生「伪漂流」，导致：
    #   - 回溯天数越多，位移反而越少（冻结帧流场方向可能相反）
    #   - 不同回溯天数的轨迹完全不同（一个在有效区间、一个已冻结）
    # 正确做法：回传 None 让呼叫端停用该粒子（STATUS_TIME_EXCEEDED）。
    total_elapsed = max(time_offset_hours, 0.0) + max(hours_elapsed, 0.0)
    max_elapsed_hours = max(n_times - 1, 0) * dt_hours
    if total_elapsed > max_elapsed_hours + 1e-9:
        return None, None

    t_exact = (n_times - 1) - total_elapsed / dt_hours
    t_exact = np.clip(t_exact, 0.0, max(n_times - 1.0001, 0.0))
    t_idx = int(np.floor(t_exact))
    t_next = min(t_idx + 1, n_times - 1)
    w = t_exact - t_idx

    spatial_index = field.get("spatial_index")
    u0 = interpolate_2d_vectorized(lon_arr, lat_arr, field["u"][t_idx], field["lat"], field["lon"], spatial_index)
    u1 = interpolate_2d_vectorized(lon_arr, lat_arr, field["u"][t_next], field["lat"], field["lon"], spatial_index)
    v0 = interpolate_2d_vectorized(lon_arr, lat_arr, field["v"][t_idx], field["lat"], field["lon"], spatial_index)
    v1 = interpolate_2d_vectorized(lon_arr, lat_arr, field["v"][t_next], field["lat"], field["lon"], spatial_index)

    u = (1 - w) * u0 + w * u1
    v = (1 - w) * v0 + w * v1

    # 僅過濾非有限值（NaN/Inf），不對原始場施加物理截斷。
    # 速度上限防禦已移至 rk4_step_2d 的「合成總速度」階段，
    # 避免 WRF 原始風場被錯誤閹割（見 WIND_SPEED_CAP_MPS）。
    u = np.where(np.isfinite(u), u, 0.0)
    v = np.where(np.isfinite(v), v, 0.0)

    return u, v

def _valid_water_point(lon, lat, prov):
    bbox = prov["curr"]["bbox"]
    if not _inside_bbox(lon, lat, bbox):
        return False
    return not is_land_vectorized(np.array([lon]), np.array([lat]), prov)[0]

def snap_to_valid_ocean(lon0, lat0, prov, max_km=AUTO_OFFSHORE_START_KM):
    """將座標吸附至最近的有效海域點。

    回傳 (lon, lat, snapped_km)：
      - snapped_km = 0.0 表示原點即為有效海域
      - snapped_km > 0   表示已移動的距離（公里）
      - snapped_km < 0   表示吸附失敗（-1.0），呼叫端應據此處理
    """
    if _valid_water_point(lon0, lat0, prov):
        return float(lon0), float(lat0), 0.0

    radii_m = np.arange(1000.0, max_km * 1000.0, AUTO_OFFSHORE_STEP_M)
    thetas = np.linspace(0, 2 * np.pi, 36, endpoint=False)

    for r in radii_m:
        for th in thetas:
            lon = lon0 + meters_to_deg_lon(r * np.cos(th), lat0)
            lat = lat0 + meters_to_deg_lat(r * np.sin(th))
            if _valid_water_point(lon, lat, prov):
                return float(lon), float(lat), float(r / 1000.0)

    # 吸附失敗：明確回報，不再靜默回傳陸地座標。
    return float(lon0), float(lat0), -1.0

def rk4_step_2d(lon, lat, dt_seconds, prov, hours_elapsed, windage, time_offset_hours=0.0):
    # time_offset_hours 可為：
    #   - 純量：curr 與 wind 共用同一偏移（兩者時間軸一致時）
    #   - dict：{"curr": x, "wind": y}，分別指定各場偏移
    if isinstance(time_offset_hours, dict):
        curr_offset = float(time_offset_hours.get("curr", 0.0))
        wind_offset = float(time_offset_hours.get("wind", 0.0))
    else:
        curr_offset = wind_offset = float(time_offset_hours)

    def to_dlon(u, lat_here):
        return u / (METERS_PER_DEG_LAT * np.cos(np.deg2rad(lat_here)) + 1e-12)
    def to_dlat(v):
        return v / METERS_PER_DEG_LAT
    def get_uv(l, la, t_elap):
        # ============================================================
        # 第一層：分量獨立 QC（Component-wise Plausibility Envelope）
        #   各物理分量先各自過濾異常格點，避免異常值污染合成速度。
        # ============================================================
        uc, vc = get_velocity_interpolated(l, la, prov["curr"], t_elap, curr_offset)
        # 時間越界：速度場已無有效資料，回傳 None 讓呼叫端停用粒子，
        # 嚴禁以凍結幀（資料第一幀）繼續積分造成偽漂流。
        if uc is None:
            return None, None
        c_spd = np.hypot(uc, vc)
        c_over = c_spd > CURRENT_SPEED_CAP_MPS
        c_scale = np.where(c_over, CURRENT_SPEED_CAP_MPS / (c_spd + 1e-12), 1.0)
        uc = uc * c_scale
        vc = vc * c_scale

        if prov["wind"] is not None:
            uw, vw = get_velocity_interpolated(l, la, prov["wind"], t_elap, wind_offset)
            if uw is None:
                return None, None
            w_spd = np.hypot(uw, vw)
            w_over = w_spd > WIND_SPEED_CAP_MPS
            w_scale = np.where(w_over, WIND_SPEED_CAP_MPS / (w_spd + 1e-12), 1.0)
            # 風場以 windage 係數耦合為表面漂流分量（非直接疊加）。
            uc = uc + uw * w_scale * windage
            vc = vc + vw * w_scale * windage

        # ============================================================
        # 第二層：合成速度動態 QC（Composite Dynamic Envelope）
        #   上限由物理分量動態推導，而非硬編碼：
        #     total_cap = V_curr_max + windage * V_wind_max
        #   例：windage=0.015 → 3.0 + 0.015*35.0 = 3.525 m/s
        #       windage=0.05  → 3.0 + 0.05 *35.0 = 4.750 m/s
        # ============================================================
        total_cap = CURRENT_SPEED_CAP_MPS + windage * WIND_SPEED_CAP_MPS
        total_spd = np.hypot(uc, vc)
        over = total_spd > total_cap
        scale = np.where(over, total_cap / (total_spd + 1e-12), 1.0)
        return uc * scale, vc * scale

    u1, v1 = get_uv(lon, lat, hours_elapsed)
    if u1 is None:
        return None, None, None, None
    dlon1 = to_dlon(u1, lat); dlat1 = to_dlat(v1)

    t2 = hours_elapsed + abs(0.5 * dt_seconds) / 3600.0
    lon2 = lon + 0.5 * dt_seconds * dlon1
    lat2 = lat + 0.5 * dt_seconds * dlat1

    u2, v2 = get_uv(lon2, lat2, t2)
    if u2 is None:
        return None, None, None, None
    dlon2 = to_dlon(u2, lat2); dlat2 = to_dlat(v2)

    lon3 = lon + 0.5 * dt_seconds * dlon2
    lat3 = lat + 0.5 * dt_seconds * dlat2

    u3, v3 = get_uv(lon3, lat3, t2)
    if u3 is None:
        return None, None, None, None
    dlon3 = to_dlon(u3, lat3); dlat3 = to_dlat(v3)

    t4 = hours_elapsed + abs(dt_seconds) / 3600.0
    lon4 = lon + dt_seconds * dlon3
    lat4 = lat + dt_seconds * dlat3

    u4, v4 = get_uv(lon4, lat4, t4)
    if u4 is None:
        return None, None, None, None
    dlon4 = to_dlon(u4, lat4); dlat4 = to_dlat(v4)

    new_lon = lon + (dt_seconds / 6.0) * (dlon1 + 2 * dlon2 + 2 * dlon3 + dlon4)
    new_lat = lat + (dt_seconds / 6.0) * (dlat1 + 2 * dlat2 + 2 * dlat3 + dlat4)

    # 回傳 RK4 路徑上的 4 個中間節點（k1/k2/k3 終點 + 終點），供觸岸判定使用。
    # 這些點在積分過程中已算出，額外回傳不增加任何計算成本。
    # 觸岸判定必須涵蓋整條路徑，否則單步位移（可達 6.3 km）大於網格間距
    # （約 2.2 km）時，粒子會「跨過」陸地格而未被偵測（穿陸 bug）。
    mid_lons = np.vstack([lon2, lon3, lon4, new_lon])
    mid_lats = np.vstack([lat2, lat3, lat4, new_lat])

    return new_lon, new_lat, mid_lons, mid_lats

# ============================================================
# VISUAL HELPERS
# ============================================================

def build_hotspot_dataframe():
    return pd.DataFrame([
        {"name": name, "lon": float(v[0]), "lat": float(v[1])}
        for name, v in HOTSPOT_CENTERS.items()
    ])

def particles_for_site(particle_sites, site_name):
    if site_name in ["All Sites", "全部熱點 (All Sites)"]:
        return np.arange(len(particle_sites), dtype=int)
    return np.array([i for i, s in enumerate(particle_sites) if s == site_name], dtype=int)

def snapshot_at_step(df, step_idx):
    d = df[df["step_index"] <= step_idx].sort_values(["particle_id", "step_index"])
    return d.groupby("particle_id", as_index=False).tail(1).copy()

def nearest_hotspot_distance_km(lons, lats):
    centers = np.array(list(HOTSPOT_CENTERS.values()), dtype=float)
    min_dist_km = np.full(len(np.asarray(lons)), np.inf, dtype=float)

    for cx, cy in centers:
        dist = haversine(lons, lats, cx, cy) / 1000.0
        min_dist_km = np.minimum(min_dist_km, dist)

    return min_dist_km

def _last_valid_snapshot(df_subset, replay_step):
    """取得每顆粒子在 [0, replay_step] 內「最後一個有效（非 NaN）位置」。

    為何需要此函式：
      粒子在回溯過程中若觸岸（STATUS_BEACHED）或越界，其座標會被設為 NaN。
      若直接對 replay_step 這一格取快照，會把「剛好在最後一步觸岸」的粒子
      整批丟棄，導致 current 為空、淨位移被誤算為 0.0（高雄 0 位移的根因）。
      正確做法是回溯每顆粒子在失效前的最後有效位置，保留其真實漂流軌跡。
    """
    upto = df_subset[df_subset["step_index"] <= replay_step].copy()
    upto = upto[np.isfinite(upto["lon"]) & np.isfinite(upto["lat"])]
    if upto.empty:
        return upto
    return upto.sort_values(["particle_id", "step_index"]).groupby(
        "particle_id", as_index=False
    ).tail(1).copy()


def bootstrap_proportion_ci(flags, n_boot=2000, ci=95.0, seed=42):
    """以 bootstrap 重抽樣估計「比例」的信賴區間。

    為何需要此函式（不確定性量化）：
      compute_replay_metrics 目前只輸出「本地源佔比」的點估計（例如 62.3%），
      並以 >= 50% 硬切結論。但點估計沒有誤差範圍，無法區分：
        (A) 62.3% ± 3%  → 結論穩健（確實 > 50%）
        (B) 62.3% ± 20% → 結論不顯著（可能 < 50%）
      本函式以 bootstrap 重抽樣（resampling with replacement）估計比例的
      95% 信賴區間，讓「本地源 / 境外源」的判定具備統計信度。

    參數
    ----
    flags : array-like of bool
        每顆粒子的二元判定（True = 本地源，False = 境外源）。
    n_boot : int
        重抽樣次數（預設 2000，足以穩定估計 95% CI）。
    ci : float
        信賴水準（預設 95.0）。
    seed : int
        隨機種子，確保結果可重現。

    回傳
    ----
    dict:
        point_pct : 點估計（百分比）
        lo_pct    : 信賴區間下界（百分比）
        hi_pct    : 信賴區間上界（百分比）
        n         : 樣本數（粒子數）
        ci        : 信賴水準
        significant : 是否顯著偏離 50%（CI 不含 50%）
    """
    arr = np.asarray(flags, dtype=bool)
    n = int(arr.size)
    if n == 0:
        return {"point_pct": 0.0, "lo_pct": 0.0, "hi_pct": 0.0,
                "n": 0, "ci": ci, "significant": False}

    point = float(arr.mean() * 100.0)

    # 樣本數極少時 bootstrap 不穩定，退化為點估計（CI = 點估計）。
    if n < 5:
        return {"point_pct": point, "lo_pct": point, "hi_pct": point,
                "n": n, "ci": ci, "significant": False}

    rng = np.random.default_rng(seed)
    # 向量化重抽樣：一次產生 (n_boot, n) 的索引矩陣，避免 Python 迴圈。
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1) * 100.0

    alpha = (100.0 - ci) / 2.0
    lo = float(np.percentile(boot_means, alpha))
    hi = float(np.percentile(boot_means, 100.0 - alpha))

    # 顯著性：95% CI 是否完全落在 50% 的同一側。
    significant = bool(lo > 50.0 or hi < 50.0)

    return {"point_pct": point, "lo_pct": lo, "hi_pct": hi,
            "n": n, "ci": ci, "significant": significant}


def _empty_replay_metrics():
    """compute_replay_metrics 的空白/退化回傳值（含不確定性量化欄位）。"""
    return {
        "net_displacement_km": 0.0,
        "total_distance_km": 0.0,
        "local_pct": 0.0,
        "external_pct": 100.0,
        "prediction": "境外源 (External Source)",
        "local_ci_lo": 0.0,
        "local_ci_hi": 0.0,
        "ci_level": 95.0,
        "n_particles": 0,
        "significant": False,
        "confidence_label": "不顯著 (Inconclusive)",
    }


def compute_replay_metrics(df_subset, replay_step, local_radius_km=LOCAL_SOURCE_RADIUS_KM_DEFAULT):
    if df_subset is None or df_subset.empty:
        return _empty_replay_metrics()

    # 使用「最後有效位置」而非嚴格快照，避免最後一步觸岸的粒子被整批丟棄。
    current = _last_valid_snapshot(df_subset, replay_step)
    start = snapshot_at_step(df_subset, 0)
    upto = df_subset[df_subset["step_index"] <= replay_step]

    if current.empty or start.empty:
        return _empty_replay_metrics()

    current = current[np.isfinite(current["lon"]) & np.isfinite(current["lat"])].copy()
    start = start[np.isfinite(start["lon"]) & np.isfinite(start["lat"])].copy()
    if current.empty or start.empty:
        return _empty_replay_metrics()

    start_centroid_lon = float(start["lon"].mean())
    start_centroid_lat = float(start["lat"].mean())
    current_centroid_lon = float(current["lon"].mean())
    current_centroid_lat = float(current["lat"].mean())

    net_displacement_km = float(haversine(start_centroid_lon, start_centroid_lat, current_centroid_lon, current_centroid_lat) / 1000.0)
    total_distance_km = float(upto.groupby("particle_id")["dist_step"].sum().mean() / 1000.0)

    nearest_km = nearest_hotspot_distance_km(current["lon"].values, current["lat"].values)

    # 觸岸判定：粒子在 [0, replay_step] 內「曾經」觸岸即視為陸源。
    # 因為 _last_valid_snapshot 取的是觸岸前一步（status=ACTIVE），
    # 若只看 current 的 status_code 會漏判所有已觸岸粒子。
    beached_ids = set(
        upto.loc[upto["status_code"] == STATUS_BEACHED, "particle_id"].unique().tolist()
    )
    is_terrestrial = current["particle_id"].isin(beached_ids).to_numpy()
    is_nearshore = (nearest_km <= local_radius_km)
    is_local = is_terrestrial | is_nearshore

    local_pct = float(is_local.mean() * 100.0)
    external_pct = float(100.0 - local_pct)

    prediction = "本地源 (Local Source)" if local_pct >= external_pct else "境外源 (External Source)"

    # 不確定性量化：以 bootstrap 估計「本地源佔比」的 95% 信賴區間。
    # 若 CI 跨越 50%，代表「本地/境外」的判定在統計上不顯著，
    # 結論應標註為「傾向不明確 (Inconclusive)」而非武斷二選一。
    ci_stats = bootstrap_proportion_ci(is_local, n_boot=2000, ci=95.0)
    if ci_stats["significant"]:
        confidence_label = "顯著 (Significant)"
    else:
        confidence_label = "不顯著 (Inconclusive)"

    return {
        "net_displacement_km": net_displacement_km,
        "total_distance_km": total_distance_km,
        "local_pct": local_pct,
        "external_pct": external_pct,
        "prediction": prediction,
        "local_ci_lo": ci_stats["lo_pct"],
        "local_ci_hi": ci_stats["hi_pct"],
        "ci_level": ci_stats["ci"],
        "n_particles": ci_stats["n"],
        "significant": ci_stats["significant"],
        "confidence_label": confidence_label,
    }

def compute_site_summary(df_ready, replay_step, local_radius_km=LOCAL_SOURCE_RADIUS_KM_DEFAULT):
    rows = []
    for site_name, site_df in df_ready.groupby("site_name"):
        stats = compute_replay_metrics(site_df, replay_step, local_radius_km=local_radius_km)
        rows.append({
            "監測熱點": site_name, 
            "淨位移 (km)": round(stats["net_displacement_km"], 1),
            "總漂流里程 (km)": round(stats["total_distance_km"], 1), 
            "本地源佔比 (%)": round(stats["local_pct"], 1),
            "95% 信賴區間": f"[{stats['local_ci_lo']:.1f}, {stats['local_ci_hi']:.1f}]",
            "統計顯著性": stats["confidence_label"],
            "境外源佔比 (%)": round(stats["external_pct"], 1), 
            "來源推論傾向": stats["prediction"]
        })
    return pd.DataFrame(rows).sort_values(by=["本地源佔比 (%)", "淨位移 (km)"], ascending=[False, True])


def run_sensitivity_sweep(param_name, param_values, base_kwargs, release_sites,
                          total_particles, total_steps, dt_mins, prov,
                          reference_time_utc=None, local_radius_km=LOCAL_SOURCE_RADIUS_KM_DEFAULT):
    """單一參數敏感度掃描 (One-at-a-Time Sensitivity Sweep)。

    為何需要此函式（模型不確定性）：
      bootstrap 信賴區間量化的是「抽樣誤差」（統計不確定性），
      但無法回答「若把風阻係數從 1.5% 改成 3.5%，結論會不會翻盤？」
      （模型不確定性）。本函式對單一參數掃描一組值，每次重跑完整模擬，
      記錄「本地源佔比」隨參數變化的曲線，用以判斷結論的穩健性。

    參數
    ----
    param_name : str
        要掃描的參數名稱，支援：
          "windage"      → 海面風阻係數 (%)
          "local_radius" → 本地源關聯半徑 (km)
          "days"         → 回溯天數（會重算 total_steps）
          "dt_mins"      → 數值積分步長（會重算 total_steps）
    param_values : list
        要測試的參數值列表。
    base_kwargs : dict
        基準參數（含 windage, days, dt_mins 等），未掃描的參數沿用此值。
    release_sites, total_particles, prov, reference_time_utc :
        與 simulate_particles 相同。
    total_steps, dt_mins :
        基準總步數與步長（掃描 days/dt_mins 時會覆寫）。

    回傳
    ----
    pd.DataFrame，欄位：
        param_value, local_pct, external_pct, net_displacement_km,
        total_distance_km, ci_lo, ci_hi, significant, prediction
    """
    rows = []
    for val in param_values:
        # 依掃描參數決定本次模擬的實際設定。
        cur_windage = base_kwargs.get("windage", 1.5)
        cur_radius = local_radius_km
        cur_steps = total_steps
        cur_dt = dt_mins

        if param_name == "windage":
            cur_windage = float(val)
        elif param_name == "local_radius":
            cur_radius = float(val)
        elif param_name == "days":
            # 回溯天數改變 → 重算總步數（步長不變）。
            cur_steps = max(1, int(round(float(val) * 24.0 * 60.0 / cur_dt)))
        elif param_name == "dt_mins":
            # 步長改變 → 在相同回溯時數下重算總步數。
            base_hours = base_kwargs.get("days", 5.0) * 24.0
            cur_dt = float(val)
            cur_steps = max(1, int(round(base_hours * 60.0 / cur_dt)))
        else:
            raise ValueError(f"不支援的敏感度參數：{param_name}")

        df, _, _, _, _ = simulate_particles(
            release_sites=release_sites,
            total_particles=total_particles,
            total_steps=cur_steps,
            dt_mins=cur_dt,
            prov=prov,
            windage=cur_windage,
            reference_time_utc=reference_time_utc,
        )

        if df is None or df.empty:
            rows.append({
                "param_value": val, "local_pct": 0.0, "external_pct": 100.0,
                "net_displacement_km": 0.0, "total_distance_km": 0.0,
                "ci_lo": 0.0, "ci_hi": 0.0, "significant": False,
                "prediction": "境外源 (External Source)",
            })
            continue

        # 重算 dist_step（simulate_particles 不回傳此欄位）。
        df = df.sort_values(by=["particle_id", "step_index"]).copy()
        df["prev_lon"] = df.groupby("particle_id")["lon"].shift(1).fillna(df["lon"])
        df["prev_lat"] = df.groupby("particle_id")["lat"].shift(1).fillna(df["lat"])
        df["dist_step"] = haversine(df["prev_lon"], df["prev_lat"], df["lon"], df["lat"])

        stats = compute_replay_metrics(df, cur_steps, local_radius_km=cur_radius)
        rows.append({
            "param_value": val,
            "local_pct": stats["local_pct"],
            "external_pct": stats["external_pct"],
            "net_displacement_km": stats["net_displacement_km"],
            "total_distance_km": stats["total_distance_km"],
            "ci_lo": stats["local_ci_lo"],
            "ci_hi": stats["local_ci_hi"],
            "significant": stats["significant"],
            "prediction": stats["prediction"],
        })

    return pd.DataFrame(rows)


# 匯出 PNG 用的中→英對照表。
# 原因：Streamlit Cloud（Linux 容器）的 kaleido/Chromium 環境缺少中文字型，
# 直接匯出會讓所有中文變成「豆腐塊」（□）。網頁顯示不受影響（瀏覽器有字型），
# 但後端渲染的 PNG 會壞掉。故匯出前先將圖表文字換成英文，確保 PNG 可讀。
_PNG_TEXT_MAP = {
    # 標題
    "漂流總里程分佈 (Travel Distance Distribution)": "Travel Distance Distribution",
    "粒子最終來源歸宿分佈 (Particle Fate / Origin Distribution)": "Particle Fate / Origin Distribution",
    # 軸標籤
    "漂流距離 (km)": "Travel Distance (km)",
    "粒子數量": "Particle Count",
    "回溯狀態類別": "Backtracking Status",
    "本地源佔比 (%)": "Local Source Fraction (%)",
    # 圖例 / 註解
    "本地源佔比": "Local Source Fraction",
    "95% 信賴區間": "95% Confidence Interval",
    "50% 判定線": "50% Decision Line",
    # 敏感度曲線標題前綴（動態組字，於下方以 replace 處理）
    "敏感度曲線：": "Sensitivity Curve: ",
    # 敏感度掃描參數名稱（動態 _label）
    "海面風阻係數 (Windage %)": "Windage (%)",
    "本地源關聯半徑 (Local Radius km)": "Local Radius (km)",
    "回溯天數 (Days)": "Backtracking Days",
    "數值積分步長 (Time Step min)": "Time Step (min)",
    # 狀態類別（legend 會用到）
    "外海漂流中 (Offshore Active)": "Offshore Active",
    "沿岸陸源釋放 (Terrestrial Origin)": "Terrestrial Origin",
    "超出模擬邊界 (Out of Bounds)": "Out of Bounds",
    "資料時間軸耗盡 (Time Exceeded)": "Time Exceeded",
}


# def _translate_fig_text(fig):
#     """回傳一個「文字已英文化」的圖表副本，供 PNG 匯出使用。

#     僅複製圖表物件並替換文字，不影響網頁上顯示的原始圖表。
#     """
#     import copy as _copy
#     f = _copy.deepcopy(fig)

#     def _tr(s):
#         if not isinstance(s, str):
#             return s
#         for zh, en in _PNG_TEXT_MAP.items():
#             if zh in s:
#                 s = s.replace(zh, en)
#         return s

    # 標題
    if f.layout.title and f.layout.title.text:
        f.layout.title.text = _tr(f.layout.title.text)
    # 軸標題
    for ax in (f.layout.xaxis, f.layout.yaxis):
        if ax is not None and ax.title and ax.title.text:
            ax.title.text = _tr(ax.title.text)
    # 圖例標題（如「回溯狀態類別」）
    if f.layout.legend and f.layout.legend.title and f.layout.legend.title.text:
        f.layout.legend.title.text = _tr(f.layout.legend.title.text)
    # 圖例名稱（traces）
    for tr in f.data:
        if getattr(tr, "name", None):
            tr.name = _tr(tr.name)
        # trace 的 x 值：類別軸（如 final_status）會把中文類別值放在這裡，
        # 這些值會顯示在 x 軸刻度上。僅在元素為字串時翻譯，避免動到數值。
        # 注意：plotly 對大型陣列會用特殊編碼（dict 形式，含 dtype/bdata），
        # 必須排除，否則會破壞圖表資料。
        xv = getattr(tr, "x", None)
        if xv is not None and not isinstance(xv, dict):
            try:
                xlist = list(xv)
                if xlist and all(isinstance(v, str) for v in xlist):
                    tr.x = tuple(_tr(v) for v in xlist)
            except (TypeError, IndexError, KeyError):
                pass
    # 軸刻度標籤（類別軸，如 final_status 的類別值）
    for ax in (f.layout.xaxis, f.layout.yaxis):
        if ax is None:
            continue
        if getattr(ax, "ticktext", None):
            ax.ticktext = tuple(_tr(t) for t in ax.ticktext)
        if getattr(ax, "categoryarray", None):
            ax.categoryarray = tuple(_tr(t) for t in ax.categoryarray)
    # 註解（如 50% 判定線）
    if f.layout.annotations:
        for ann in f.layout.annotations:
            if getattr(ann, "text", None):
                ann.text = _tr(ann.text)
    return f


# def fig_to_png_bytes(fig, width=1400, height=800, scale=2):
#     """將 plotly 圖表轉為 PNG 位元組，供 st.download_button 下載。

#     為何需要此函式（研究成果匯出）：
#       復賽報告「研究成果」需附具體圖表。使用者可直接從系統匯出高解析度
#       PNG（scale=2 即 2 倍解析度，適合列印），無須手動截圖。

#     中文處理：雲端 kaleido/Chromium 缺中文字型，故匯出前先將圖表文字
#       英文化（見 _translate_fig_text），避免 PNG 出現「豆腐塊」。

#     依賴：kaleido（plotly 靜態圖片引擎）。若未安裝則回傳 None，
#     呼叫端應顯示提示而非崩潰。
#     """
#     try:
#         export_fig = _translate_fig_text(fig)
#         return export_fig.to_image(format="png", width=width, height=height, scale=scale)
#     except Exception:
#         return None


# def render_figure_export(fig, filename, label="圖表", key=None):
#     """在 Streamlit 中渲染「下載此圖 PNG」按鈕。

#     若 kaleido 不可用，顯示提示訊息（不中斷頁面）。
#     """
#     png_bytes = fig_to_png_bytes(fig)
#     if png_bytes is None:
#         st.caption("⚠️ 圖表匯出需安裝 `kaleido`（`pip install kaleido`），目前無法產生 PNG。")
#         return
#     st.download_button(
#         f"⬇️ 下載{label} PNG (高解析度)",
#         data=png_bytes,
#         file_name=filename,
#         mime="image/png",
#         use_container_width=True,
#         key=key,
#     )

def build_paths_from_history(history_lon, history_lat, step_idx, particle_indices, particle_sites=None, site_kind_map=None, max_particles=200):
    if step_idx < 1 or len(particle_indices) == 0:
        return []

    if len(particle_indices) > max_particles:
        rng = np.random.default_rng(42)
        particle_indices = rng.choice(particle_indices, max_particles, replace=False)

    paths = []
    for pid in particle_indices:
        path = []
        # 逐格累積軌跡；遇到 NaN（粒子觸岸/越界/時間耗盡）時「停止累積」，
        # 但保留 NaN 之前已走過的有效軌跡，而非丟棄整條路徑。
        # 舊邏輯遇到 NaN 直接 break 並 continue，導致「最後一步才觸岸」的粒子
        # 整條軌跡消失（地圖上只剩起點圓點、看不到任何軌跡線）。
        for i in range(step_idx + 1):
            lon_i = float(history_lon[i][pid])
            lat_i = float(history_lat[i][pid])
            if not np.isfinite(lon_i) or not np.isfinite(lat_i):
                break
            path.append([lon_i, lat_i])

        # 至少要有兩個點才能構成線段；單點或空路徑略過。
        if len(path) < 2:
            continue

        site_name = None
        kind = "境外源 (External Source)"

        if particle_sites is not None and pid < len(particle_sites):
            site_name = particle_sites[pid]

        if site_kind_map is not None and site_name in site_kind_map:
            kind = site_kind_map[site_name]

        paths.append({"path": path, "color": source_kind_to_path_color(kind)})

    return paths

def render_trajectory_map(history_lon, history_lat, step_idx, particle_indices, particle_sites=None, site_kind_map=None):
    render_map_legend()

    paths = build_paths_from_history(history_lon, history_lat, step_idx, particle_indices, particle_sites, site_kind_map, max_particles=200)

    current_lon = np.asarray(history_lon[step_idx], dtype=float)
    current_lat = np.asarray(history_lat[step_idx], dtype=float)

    if len(particle_indices) == 0:
        particle_indices = np.arange(len(current_lon), dtype=int)

    # 圓點位置：優先取 step_idx 當格座標；若該粒子已在此格失效（NaN），
    # 則回溯取其「最後一個有效位置」，避免觸岸粒子在地圖上憑空消失。
    dot_lon = np.full(len(particle_indices), np.nan, dtype=float)
    dot_lat = np.full(len(particle_indices), np.nan, dtype=float)
    for k, pid in enumerate(particle_indices):
        for i in range(step_idx, -1, -1):
            lon_i = float(history_lon[i][pid])
            lat_i = float(history_lat[i][pid])
            if np.isfinite(lon_i) and np.isfinite(lat_i):
                dot_lon[k] = lon_i
                dot_lat[k] = lat_i
                break

    current_df = pd.DataFrame({
        "lon": dot_lon,
        "lat": dot_lat,
        "site_name": [particle_sites[i] if particle_sites is not None and i < len(particle_sites) else "" for i in particle_indices]
    })
    current_df = current_df[np.isfinite(current_df["lon"]) & np.isfinite(current_df["lat"])].copy()

    if site_kind_map is None:
        site_kind_map = {}

    current_df["kind"] = current_df["site_name"].map(site_kind_map).fillna("境外源 (External Source)")
    current_df["color"] = current_df["kind"].apply(source_kind_to_color)

    hotspot_df = build_hotspot_dataframe()
    hotspot_df["radius"] = HOTSPOT_RADIUS

    layer_hotspots = pdk.Layer(
        "ScatterplotLayer",
        data=hotspot_df, get_position=["lon", "lat"],
        get_fill_color=HOTSPOT_FILL_COLOR, get_line_color=HOTSPOT_EDGE_COLOR,
        stroked=True, filled=True, line_width_min_pixels=2,
        get_radius="radius", radius_scale=1, radius_min_pixels=8
    )

    layer_path = pdk.Layer(
        "PathLayer", data=paths, get_path="path", get_color="color", width_min_pixels=PATH_WIDTH_MIN_PIXELS
    )

    layer_particles = pdk.Layer(
        "ScatterplotLayer",
        data=current_df, get_position=["lon", "lat"], get_fill_color="color",
        get_line_color=[255, 255, 255, 180], stroked=True, filled=True,
        line_width_min_pixels=1, get_radius=CURRENT_POINT_RADIUS, radius_scale=1, radius_min_pixels=2
    )

    center_lat = 23.5 if len(current_df) == 0 else float(current_df["lat"].mean())
    center_lon = 121.0 if len(current_df) == 0 else float(current_df["lon"].mean())

    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=7, pitch=0, bearing=0)

    st.pydeck_chart(pdk.Deck(layers=[layer_hotspots, layer_path, layer_particles], initial_view_state=view_state, map_style="dark"))

def render_heatmap(final_df):
    render_map_legend()

    final_df = final_df[
        np.isfinite(final_df["lon"]) & np.isfinite(final_df["lat"])
    ].copy()
    heat_layer = pdk.Layer("HeatmapLayer", data=final_df, get_position=["lon", "lat"], opacity=0.8, radiusPixels=60)

    hotspot_df = build_hotspot_dataframe()
    hotspot_df["radius"] = HOTSPOT_RADIUS

    hotspot_layer = pdk.Layer(
        "ScatterplotLayer", data=hotspot_df, get_position=["lon", "lat"],
        get_fill_color=HOTSPOT_FILL_COLOR, get_line_color=HOTSPOT_EDGE_COLOR,
        stroked=True, filled=True, line_width_min_pixels=2,
        get_radius="radius", radius_scale=1, radius_min_pixels=8
    )

    center_lat = 23.5 if len(final_df) == 0 else float(final_df["lat"].mean())
    center_lon = 121.0 if len(final_df) == 0 else float(final_df["lon"].mean())

    heat_view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6, pitch=0, bearing=0)

    st.pydeck_chart(pdk.Deck(layers=[hotspot_layer, heat_layer], initial_view_state=heat_view, map_style="dark"))

# ============================================================
# SIMULATION
# ============================================================

def simulate_particles(release_sites, total_particles, total_steps, dt_mins, prov, windage,
                       reference_time_utc=None):
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    rng = np.random.default_rng(42)

    if not release_sites:
        return pd.DataFrame(), [[]], [[]], run_id, []

    base = total_particles // len(release_sites)
    rem = total_particles % len(release_sites)
    counts = [base + (1 if i < rem else 0) for i in range(len(release_sites))]

    particle_sites = []
    lons_list = []
    lats_list = []
    snap_failures = []

    for idx, (site_name, lon0, lat0) in enumerate(release_sites):
        n_site = counts[idx]
        if n_site <= 0:
            continue

        lon0, lat0, snapped_km = snap_to_valid_ocean(lon0, lat0, prov)
        if snapped_km < 0:
            # 吸附失敗：記錄並跳過此站點，避免產生「陸地靜止粒子」
            # 造成位移恆為 0 的假象。
            snap_failures.append(site_name)
            continue

        ang = rng.random(n_site) * 2 * np.pi
        rs = np.sqrt(rng.random(n_site)) * 500.0
        dx = rs * np.cos(ang)
        dy = rs * np.sin(ang)

        lons = lon0 + meters_to_deg_lon(dx, lat0)
        lats = lat0 + meters_to_deg_lat(dy)

        particle_sites.extend([site_name] * n_site)
        lons_list.append(lons)
        lats_list.append(lats)

    if snap_failures:
        st.warning(
            "以下熱點無法吸附至有效海域，已略過（請檢查座標或海陸遮罩）："
            + "、".join(snap_failures)
        )

    lons = np.concatenate(lons_list) if lons_list else np.array([], dtype=np.float32)
    lats = np.concatenate(lats_list) if lats_list else np.array([], dtype=np.float32)

    active = np.ones(len(lons), dtype=bool)
    status_codes = np.zeros(len(lons), dtype=np.int8)

    history_lon = [lons.copy()]
    history_lat = [lats.copy()]
    history_status = [status_codes.copy()]

    rows = []
    dt_seconds = -abs(dt_mins) * 60.0
    base_time_utc = reference_time_utc or prov.get("reference_time_utc") or datetime.now(timezone.utc)

    # ------------------------------------------------------------
    # 起算時刻偏移（Time Offset）
    # ------------------------------------------------------------
    # 逆向追溯必須從「使用者選定的起算時刻」開始取速度場，
    # 而非永遠從資料最後一幀開始。此處計算起算時刻距各場
    # 「資料末端」的小時數，作為 get_velocity_interpolated 的基準偏移。
    # POM 與 WRF 時間軸不同，需各自以自己的 time_end_utc 計算。
    def _offset_hours(field):
        if field is None:
            return 0.0
        end = field.get("time_end_utc")
        if end is None:
            return 0.0
        return max(0.0, (end - base_time_utc).total_seconds() / 3600.0)

    curr_offset_hours = _offset_hours(prov.get("curr"))
    wind_offset_hours = _offset_hours(prov.get("wind"))

    for pid in range(len(lons)):
        rows.append({
            "particle_id": pid, "run_id": run_id, "site_name": particle_sites[pid],
            "lon": float(lons[pid]), "lat": float(lats[pid]),
            "step_index": 0, "time": base_time_utc,
            "status": STATUS_LABELS[STATUS_ACTIVE], "status_code": STATUS_ACTIVE
        })

    for step in range(total_steps):
        active_idx = np.where(active)[0]
        if len(active_idx) > 0:
            hours_elapsed = max(0.0, abs(step * dt_seconds) / 3600.0)
            # 可用回溯時數 = 資料總跨度 - 起算時刻偏移。
            # 起算時刻越早（offset 越大），能往前回溯的時數越少，
            # 否則會取到資料起點之前的時間（被 clip 成凍結幀 → 偽漂流）。
            #
            # 邊界必須與 get_velocity_interpolated 的越界判定完全一致：
            #   資料總跨度 = (n_times - 1) * dt_hours
            #   越界條件   = offset + elapsed > 資料總跨度
            # 舊寫法用 max(..., 0.0) 會讓 max_hours 在 offset 過大時變成 0，
            # 使 time_exceeded 恆為 False，粒子在凍結幀中一路跑到底。
            curr_span_hours = max(prov["curr"]["n_times"] - 1, 0) * prov["curr"]["dt_hours"]
            max_hours = curr_span_hours - curr_offset_hours

            # 第二道防線：若積分時間超出資料覆蓋時限，標記為 STATUS_TIME_EXCEEDED 並停止，
            # 嚴禁將 hours_elapsed 夾在 max_hours 造成「時間凍結下的偽漂流」。
            time_exceeded = hours_elapsed > max_hours + 1e-9
            if time_exceeded:
                exceeded_idx = active_idx
                status_codes[exceeded_idx] = STATUS_TIME_EXCEEDED
                lons[exceeded_idx] = np.nan
                lats[exceeded_idx] = np.nan
                active[exceeded_idx] = False

                step_time = base_time_utc - timedelta(minutes=(step + 1) * abs(dt_mins))
                for pid in exceeded_idx:
                    rows.append({
                        "particle_id": int(pid), "run_id": run_id, "site_name": particle_sites[pid],
                        "lon": np.nan, "lat": np.nan,
                        "step_index": step + 1, "time": step_time,
                        "status": STATUS_LABELS[STATUS_TIME_EXCEEDED], "status_code": STATUS_TIME_EXCEEDED
                    })

                history_lon.append(lons.copy())
                history_lat.append(lats.copy())
                history_status.append(status_codes.copy())
                continue

            lon_sub, lat_sub, mid_lons, mid_lats = rk4_step_2d(
                lons[active_idx], lats[active_idx],
                dt_seconds, prov, hours_elapsed, windage,
                time_offset_hours={"curr": curr_offset_hours, "wind": wind_offset_hours},
            )

            # 時間越界（速度場已無有效資料）：rk4_step_2d 回傳 None。
            # 此時必須停用這批粒子，絕不可用凍結幀繼續積分。
            if lon_sub is None:
                status_codes[active_idx] = STATUS_TIME_EXCEEDED
                lons[active_idx] = np.nan
                lats[active_idx] = np.nan
                active[active_idx] = False

                step_time = base_time_utc - timedelta(minutes=(step + 1) * abs(dt_mins))
                for pid in active_idx:
                    rows.append({
                        "particle_id": int(pid), "run_id": run_id, "site_name": particle_sites[pid],
                        "lon": np.nan, "lat": np.nan,
                        "step_index": step + 1, "time": step_time,
                        "status": STATUS_LABELS[STATUS_TIME_EXCEEDED], "status_code": STATUS_TIME_EXCEEDED
                    })

                history_lon.append(lons.copy())
                history_lat.append(lats.copy())
                history_status.append(status_codes.copy())
                continue

            bbox = prov["curr"]["bbox"]
            out_of_bounds = ((lon_sub < bbox[0]) | (lon_sub > bbox[1]) | (lat_sub < bbox[2]) | (lat_sub > bbox[3]))

            # 觸岸判定涵蓋整條 RK4 路徑（終點 + 4 個中間節點），
            # 避免單步位移大於網格間距時「跨過」陸地格而漏判（穿陸 bug）。
            beached = is_land_vectorized(lon_sub, lat_sub, prov)
            for _k in range(mid_lons.shape[0]):
                beached |= is_land_vectorized(mid_lons[_k], mid_lats[_k], prov)

            next_active = ~(out_of_bounds | beached)

            lons[active_idx] = lon_sub
            lats[active_idx] = lat_sub

            if np.any(out_of_bounds):
                status_codes[active_idx[out_of_bounds]] = STATUS_OUT_OF_BOUNDS
            if np.any(beached):
                status_codes[active_idx[beached]] = STATUS_BEACHED

            dead_idx = active_idx[~next_active]
            if len(dead_idx) > 0:
                lons[dead_idx] = np.nan
                lats[dead_idx] = np.nan

            active[active_idx] = next_active

            step_time = base_time_utc - timedelta(minutes=(step + 1) * abs(dt_mins))

            for k, pid in enumerate(active_idx):
                st_code = STATUS_ACTIVE if next_active[k] else (STATUS_BEACHED if beached[k] else STATUS_OUT_OF_BOUNDS)
                rows.append({
                    "particle_id": int(pid), "run_id": run_id, "site_name": particle_sites[pid],
                    # 失效粒子的最后一笔也不保留陆地/越界坐标，
                    # 避免 replay metrics 把无效终点当成有效样本。
                    "lon": float(lon_sub[k]) if next_active[k] else np.nan,
                    "lat": float(lat_sub[k]) if next_active[k] else np.nan,
                    "step_index": step + 1, "time": step_time,
                    "status": STATUS_LABELS[st_code], "status_code": st_code
                })

        history_lon.append(lons.copy())
        history_lat.append(lats.copy())
        history_status.append(status_codes.copy())

    df = pd.DataFrame(rows)
    return df, history_lon, history_lat, run_id, particle_sites

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("模擬參數配置 (Simulation Setup)")

st.sidebar.divider()
st.sidebar.subheader("1. 追溯起始熱點 (Release Site)")
_ALL_SITES_OPTION = "🌏 全部熱點 (All Sites)"
site_options = list(HOTSPOT_CENTERS.keys()) + [_ALL_SITES_OPTION, "自訂座標 (Custom)"]
site_name = st.sidebar.selectbox(
    "選擇目標海岸熱點 (Target Hotspot)",
    site_options,
    help=(
        "選擇要追溯的目標海岸熱點。\n\n"
        "• 單一熱點：只針對該點釋放粒子並回溯，速度最快。\n"
        "• 🌏 全部熱點：同時對全部 12 個熱點各釋放粒子並回溯，"
        "可一次比較全台各地的海廢來源傾向（計算時間約為單點的 12 倍）。\n"
        "• 自訂座標：手動輸入經緯度。"
    ),
)

# 是否為「全台熱點同步反演」模式
backtrack_all_hotspots = site_name == _ALL_SITES_OPTION

if site_name == "自訂座標 (Custom)":
    init_lon = st.sidebar.number_input("經度 (Longitude)", value=120.2750, format="%.4f")
    init_lat = st.sidebar.number_input("緯度 (Latitude)", value=22.6450, format="%.4f")
elif backtrack_all_hotspots:
    # 全台模式：錨點僅為代表值，實際會對所有熱點釋放粒子
    init_lon, init_lat = HOTSPOT_CENTERS[list(HOTSPOT_CENTERS.keys())[0]]
else:
    init_lon, init_lat = HOTSPOT_CENTERS[site_name]

if backtrack_all_hotspots:
    st.sidebar.caption(f"🌏 全台模式：將對全部 {len(HOTSPOT_CENTERS)} 個熱點同步釋放粒子")
else:
    st.sidebar.caption(f"觀測錨點座標 (Anchor): {init_lat:.4f}°N, {init_lon:.4f}°E")

st.sidebar.divider()
st.sidebar.subheader("2. 時間維度 (Time Dimension)")
days = st.sidebar.slider("回溯模擬天數 (Days)", min_value=1, max_value=30, value=5)

time_step_options = {"30 分鐘 (細緻) / 30 min (Fine)": 30.0, "1 小時 (標準) / 1 hr (Standard)": 60.0, "2 小時 (快速) / 2 hr (Fast)": 120.0, "3 小時 (宏觀) / 3 hr (Coarse)": 180.0}
selected_step_label = st.sidebar.selectbox("數值積分步長 (Time Step)", list(time_step_options.keys()), index=1)
dt_mins = time_step_options[selected_step_label]

# 註：total_steps 於時間視窗夾取後才計算（見下方），
# 以確保步數與實際回溯時數一致。

st.sidebar.divider()
st.sidebar.subheader("3. 本地源關聯半徑 (Local Source Radius)")
local_source_radius_km = st.sidebar.slider(
    "近岸領海滯留判定半徑 (km) / Retention Radius (km)",
    min_value=LOCAL_SOURCE_RADIUS_KM_MIN,
    max_value=LOCAL_SOURCE_RADIUS_KM_MAX,
    value=LOCAL_SOURCE_RADIUS_KM_DEFAULT,
    step=0.1,
    help=(
        "僅用於判定「回溯期滿仍漂浮於近岸領海滯留帶」之粒子；"
        "觸岸粒子已由 STATUS_BEACHED 精確捕獲，不受此半徑影響。"
    ),
)
st.sidebar.caption("預設 22.2 km (12 浬法定領海責任水域)")

st.sidebar.divider()
st.sidebar.subheader("4. 漂流動力學物理設置 (Drift Dynamics)")
n_particles = st.sidebar.slider("模擬粒子總數 (Ensemble Size)", 50, 1000, 300, step=50)

# 材質預設快捷鍵
material_preset = st.sidebar.selectbox(
    "海廢材質特性預設 (Material Preset)",
    [
        "寶特瓶 (風阻 1.5%) / PET Bottle",
        "保麗龍 (風阻 3.5%) / EPS Foam",
        "漁網 (風阻 0.2%) / Fishing Net",
        "自訂風阻係數 (Custom)"
    ]
)

preset_windage_map = {
    "寶特瓶 (風阻 1.5%) / PET Bottle": 1.5,
    "保麗龍 (風阻 3.5%) / EPS Foam": 3.5,
    "漁網 (風阻 0.2%) / Fishing Net": 0.2,
}

default_w = preset_windage_map.get(material_preset, 1.5)
windage_percent = st.sidebar.slider(
    "海面風阻係數 (Windage %)", 
    min_value=0.0, max_value=5.0, 
    value=default_w, step=0.1,
    help="反映海廢露出水面的受風面積。保麗龍/空桶通常為 3~5%，寶特瓶為 1~2%，完全沒入水中之物體接近 0%。"
)
windage = windage_percent / 100.0

need_wind = windage > 0.0

# 嚴格載入 NODASS 數據庫
# 注意：不傳入回溯天數，確保 @st.cache_resource 快取鍵固定，
# 使用者調整回溯天數時不會觸發全量重新載入。
providers = build_velocity_providers(need_wind=need_wind)

_landmask_conf = providers['curr'].get('landmask_confidence', 'unknown')
if _landmask_conf == "low":
    st.sidebar.warning("海陸遮罩為低信度 fallback（幾何與官方遮罩皆缺失），陸地判定可能不準確。")
elif _landmask_conf == "medium":
    st.sidebar.info("海陸遮罩採用 POM 官方 mask（幾何多邊形不可用）。")

default_reference_time = providers.get("reference_time_utc", datetime.now(timezone.utc))

# ------------------------------------------------------------
# 回溯起算時刻選擇（僅選日期，自動對齊資料最新整點）
# ------------------------------------------------------------
# 設計原則：POM 為 1 小時步長、WRF 為 6 小時步長，資料本身即為整點對齊。
# 讓使用者自由輸入任意分鐘並無物理意義（相鄰分鐘會插值到同一對時間片）。
# 因此僅提供「日期」選擇，時刻自動對齊該日資料實際覆蓋的最後一個整點。
st.sidebar.divider()
st.sidebar.subheader("5. 起始時間 (Start Time)")
_time_start_candidates = [providers["curr"].get("time_start_utc")]
_time_end_candidates = [providers["curr"].get("time_end_utc")]
if providers.get("wind") is not None:
    _time_start_candidates.append(providers["wind"].get("time_start_utc"))
    _time_end_candidates.append(providers["wind"].get("time_end_utc"))
_avail_start = max(t for t in _time_start_candidates if t is not None)
_avail_end = min(t for t in _time_end_candidates if t is not None)

# 可選日期範圍（以 UTC 日期為準）
_date_min = _avail_start.date()
_date_max = _avail_end.date()

reference_date = st.sidebar.date_input(
    "回溯起算日期 (Start Date, UTC)",
    value=default_reference_time.date(),
    min_value=_date_min,
    max_value=_date_max,
    help="僅需選擇日期；時刻會自動對齊該日資料實際覆蓋的最後一個整點。",
)

# 對齊：取該日期中 <= 資料末端 的最後一個整點；若該日無資料則直接報錯。
_curr_times = providers["curr"].get("times_utc")
if _curr_times is None:
    _t0 = providers["curr"]["time_start_utc"]
    _n = providers["curr"]["n_times"]
    _dh = providers["curr"]["dt_hours"]
    _curr_times = [_t0 + timedelta(hours=_dh * i) for i in range(_n)]

_day_times = [t for t in _curr_times
              if t.date() == reference_date and _avail_start <= t <= _avail_end]
if _day_times:
    reference_time_utc = _day_times[-1]
else:
    # 該日期無有效資料：直接報錯，不自動對齊到其他日期（避免隱式平移）。
    st.sidebar.error(
        f"所選日期 {reference_date:%Y-%m-%d} 無有效資料。\n\n"
        f"可用資料範圍：{_avail_start:%Y-%m-%d %H:%M} 至 "
        f"{_avail_end:%Y-%m-%d %H:%M} UTC。\n\n"
        "請改選資料範圍內的日期後重試。"
    )
    st.stop()

time_start_candidates = [providers["curr"].get("time_start_utc")]
time_end_candidates = [providers["curr"].get("time_end_utc")]
if providers.get("wind") is not None:
    time_start_candidates.append(providers["wind"].get("time_start_utc"))
    time_end_candidates.append(providers["wind"].get("time_end_utc"))
available_start = max(t for t in time_start_candidates if t is not None)
available_end = min(t for t in time_end_candidates if t is not None)

# ------------------------------------------------------------
# 時間視窗嚴格檢查（Strict Time Window）
# ------------------------------------------------------------
# 設計原則：使用者選定的時間若超出資料範圍，**直接報錯並停止**，
# 絕不自動平移或縮短回溯區間。理由：
#   1. 自動平移會讓使用者以為跑的是自己選的時間，實際卻被偷偷改動，
#      造成結果無法對應、難以察覺的錯誤。
#   2. 自動縮短會讓「回溯天數」與實際模擬長度不符，物理意義失真。
# 因此改為明確報錯，要求使用者自行調整日期或回溯天數。
#
# 核心概念：回溯區間 = [simulation_start, reference_time_utc]，
#           長度固定為 days 天，必須完整落入 [available_start, available_end]。
#
# 檢查規則（任一不符即報錯停止）：
#   1. 共同資料範圍必須有效（available_start < available_end）
#   2. 回溯天數不得超過資料總長度
#   3. 起算時刻（reference_time_utc）不得晚於資料末端
#   4. 回溯起點（simulation_start）不得早於資料開端
# ------------------------------------------------------------
_requested_days = float(days)
_requested_hours = _requested_days * 24.0

_available_hours = (available_end - available_start).total_seconds() / 3600.0

if _available_hours <= 0:
    st.sidebar.error(
        "POM/WRF 共同資料範圍無效（起點不早於終點），無法進行模擬。"
        f"共同範圍：{available_start:%Y-%m-%d %H:%M} 至 "
        f"{available_end:%Y-%m-%d %H:%M} UTC。"
    )
    st.stop()

# 回溯區間（依使用者選定，不做任何平移）
simulation_start = reference_time_utc - timedelta(hours=_requested_hours)

# 檢查 2：回溯天數超過資料總長度
if _requested_hours > _available_hours + 1e-9:
    st.sidebar.error(
        f"回溯 {_requested_days:.1f} 天（{_requested_hours:.0f} 小時）超過資料總長度 "
        f"{_available_hours:.1f} 小時（{_available_hours/24.0:.2f} 天）。\n\n"
        f"可用資料範圍：{available_start:%Y-%m-%d %H:%M} 至 "
        f"{available_end:%Y-%m-%d %H:%M} UTC。\n\n"
        "請縮短回溯天數後重試。"
    )
    st.stop()

# 檢查 3：起算時刻晚於資料末端
if reference_time_utc > available_end:
    st.sidebar.error(
        f"回溯起算時間 {reference_time_utc:%Y-%m-%d %H:%M} UTC 晚於資料末端 "
        f"{available_end:%Y-%m-%d %H:%M} UTC。\n\n"
        "請將起算時間調整至資料範圍內後重試。"
    )
    st.stop()

# 檢查 4：回溯起點早於資料開端
if simulation_start < available_start:
    st.sidebar.error(
        f"回溯 {_requested_days:.1f} 天會使起點 {simulation_start:%Y-%m-%d %H:%M} UTC "
        f"早於資料開端 {available_start:%Y-%m-%d %H:%M} UTC。\n\n"
        f"可用資料範圍：{available_start:%Y-%m-%d %H:%M} 至 "
        f"{available_end:%Y-%m-%d %H:%M} UTC。\n\n"
        "請縮短回溯天數或將起算時間往後調整後重試。"
    )
    st.stop()

# 依「實際回溯時數」計算總積分步數，確保步數與時間跨度一致。
_effective_hours = (reference_time_utc - simulation_start).total_seconds() / 3600.0
total_steps = max(1, int(round(_effective_hours * 60.0 / dt_mins)))

if st.sidebar.button("開始反向推演 (Start Simulation)", use_container_width=True):
    release_sites = [(name, *HOTSPOT_CENTERS[name]) for name in HOTSPOT_CENTERS.keys()] if backtrack_all_hotspots else [(site_name, init_lon, init_lat)]

    with st.spinner("正在進行拉格朗日逆向數值積分計算..."):
        df, hist_lon, hist_lat, run_id, particle_sites = simulate_particles(
            release_sites=release_sites, total_particles=n_particles,
            total_steps=total_steps, dt_mins=dt_mins, prov=providers, windage=windage,
            reference_time_utc=reference_time_utc
        )

    st.session_state["df"] = df
    st.session_state["history_lon"] = hist_lon
    st.session_state["history_lat"] = hist_lat
    st.session_state["run_id"] = run_id
    st.session_state["dt_mins"] = dt_mins
    st.session_state["particle_sites"] = particle_sites
    st.session_state["reference_time_utc"] = reference_time_utc

# ============================================================
# RESULT PREP
# ============================================================

df_ready = None
history_lon = None
history_lat = None
particle_sites = None

if "df" in st.session_state:
    df_ready = st.session_state["df"].copy()
    history_lon = st.session_state["history_lon"]
    history_lat = st.session_state["history_lat"]
    particle_sites = st.session_state.get("particle_sites", [])

    df_ready["time"] = pd.to_datetime(df_ready["time"])
    df_ready = df_ready.sort_values(by=["particle_id", "step_index"])
    df_ready["prev_lon"] = df_ready.groupby("particle_id")["lon"].shift(1).fillna(df_ready["lon"])
    df_ready["prev_lat"] = df_ready.groupby("particle_id")["lat"].shift(1).fillna(df_ready["lat"])
    df_ready["dist_step"] = haversine(df_ready["prev_lon"], df_ready["prev_lat"], df_ready["lon"], df_ready["lat"])
    df_ready["speed_mps"] = df_ready["dist_step"] / (abs(st.session_state["dt_mins"]) * 60.0)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "溯源推演 (Trajectory & Heatmap)",
    "群體統計分析 (Ensemble Analytics)",
    "敏感度分析 (Sensitivity Analysis)",
])

# ============================================================
# TAB 1 - 溯源推演
# ============================================================
with tab1:
    if df_ready is None:
        st.info("請設定左側參數並點擊「開始反向推演」以載入模擬結果。")
    else:
        # 單點與批次過濾器自動適配
        if backtrack_all_hotspots:
            c_filter1, c_filter2 = st.columns([2, 2])
            site_view_options = ["全部熱點 (All Sites)"] + sorted(df_ready["site_name"].dropna().unique().tolist())
            site_view = c_filter1.selectbox("熱點聚焦檢視 (Site Filter)", site_view_options)
            map_view_mode = c_filter2.radio("空間圖層模式 (Layer Mode)", ["動態軌跡回溯 (Trajectory)", "來源密度熱圖 (Heatmap)"], horizontal=True)
        else:
            site_view = site_name
            c_info, c_filter2 = st.columns([2, 2])
            c_info.markdown(f"<div style='padding-top:12px; font-size:15px;'>📍 當前聚焦熱點：<b>{site_name}</b></div>", unsafe_allow_html=True)
            map_view_mode = c_filter2.radio("空間圖層模式 (Layer Mode)", ["動態軌跡回溯 (Trajectory)", "來源密度熱圖 (Heatmap)"], horizontal=True)
        if particle_sites:
            selected_indices = np.arange(len(particle_sites), dtype=int) if site_view in ["All Sites", "全部熱點 (All Sites)"] else particles_for_site(particle_sites, site_view)
        else:
            selected_indices = np.arange(df_ready["particle_id"].nunique(), dtype=int)

        plot_df = df_ready if site_view in ["All Sites", "全部熱點 (All Sites)"] else df_ready[df_ready["site_name"] == site_view].copy()

        # 時間軸滑桿
        replay_max = len(history_lon) - 1
        replay_step = st.slider("歷史時間軸回溯步數 (Time Step Slider)", min_value=0, max_value=replay_max, value=replay_max, step=1)

        # 真實歷史時間精準換算為臺灣時間 (UTC+8) 與 UTC
        step_records = plot_df[plot_df["step_index"] == replay_step]
        if not step_records.empty:
            curr_time_utc = pd.to_datetime(step_records["time"].iloc[0])
            curr_time_tw = curr_time_utc + pd.Timedelta(hours=8)
            curr_time_str = curr_time_tw.strftime("%Y-%m-%d %H:%M")
            utc_str = curr_time_utc.strftime("%Y-%m-%d %H:%M UTC")
        else:
            # 該熱點的粒子在此步已全部停止（觸岸/越界/逾時），故無座標記錄。
            curr_time_str = "--"
            utc_str = "--"

        st.markdown(
            f"""
            <div style="margin-top:-8px; margin-bottom:14px; font-size:14px; color:#cbd5e0;">
                ⏱ <b>當前推演歷史時刻：</b><span style="color:#63b3ed; font-weight:bold; font-size:15px;">{curr_time_str} (臺灣時間 UTC+8)</span> 
                <span style="color:#a0aec0; font-size:12px;">[{utc_str}]</span>
            </div>
            """, 
            unsafe_allow_html=True
        )

        # 核心 KPI 評估指標
        metrics = compute_replay_metrics(plot_df, replay_step, local_radius_km=local_source_radius_km)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("淨位移 (Net Displacement)", f"{metrics['net_displacement_km']:.1f} km")
        c2.metric("平均漂流總里程", f"{metrics['total_distance_km']:.1f} km")
        c3.metric("本地源機率 (Local Source)", f"{metrics['local_pct']:.1f}%")
        c4.metric(
            "來源判定信度 (Confidence)",
            metrics["confidence_label"],
            help=(
                f"本地源佔比 95% CI = "
                f"[{metrics['local_ci_lo']:.1f}%, {metrics['local_ci_hi']:.1f}%]\n"
                "顯著：CI 完全落在 50% 同一側；不顯著：CI 跨越 50%。"
            ),
        )

        site_kind_map = {}
        if site_view in ["All Sites", "全部熱點 (All Sites)"]:
            for s in sorted(plot_df["site_name"].dropna().unique().tolist()):
                site_kind_map[s] = compute_replay_metrics(
                    plot_df[plot_df["site_name"] == s], replay_step,
                    local_radius_km=local_source_radius_km
                )["prediction"]
        else:
            site_kind_map[site_view] = metrics["prediction"]

        # 圖層切換呈現
        if map_view_mode == "動態軌跡回溯 (Trajectory)":
            render_trajectory_map(history_lon, history_lat, replay_step, selected_indices, particle_sites, site_kind_map)
        else:
            # 熱圖快照須取每顆粒子「最後一個有效（非 NaN）位置」：
            # 觸岸/越界粒子在失效當格的 lon/lat 為 NaN，若用 snapshot_at_step
            # 會把這些粒子整批丟棄，導致部分熱點（粒子多已觸岸）熱圖空白。
            current_snapshot = _last_valid_snapshot(plot_df, replay_step)
            render_heatmap(current_snapshot)

        # 全台熱點對比分析表
        if backtrack_all_hotspots:
            st.subheader("全台熱點海廢來源傾向分析 (Site-wise Assessment)")
            st.dataframe(compute_site_summary(plot_df, replay_step, local_radius_km=local_source_radius_km), use_container_width=True, height=260)

        # 收合式數據匯出器
        with st.expander("匯出本次模擬完整數值結果 (Export Dataset CSV)"):
            csv_data = df_ready.to_csv(index=False).encode("utf-8")
            st.download_button("下載完整軌跡 CSV (Download CSV)", csv_data, "nodass_backtracking_results.csv", use_container_width=True)
            st.dataframe(df_ready.head(100), use_container_width=True, height=220)

# ============================================================
# TAB 2 - 群體統計分析
# ============================================================
with tab2:
    if df_ready is None:
        st.info("請先執行模擬以檢視群體統計指標。")
    else:
        st.subheader("漂流動力學群體統計 (Ensemble Analytics)")

        pids = df_ready["particle_id"].unique()
        dist_rows = []
        for pid in pids:
            d = df_ready[df_ready["particle_id"] == pid].sort_values("step_index")
            # 粒子觸岸後座標會被設為 NaN；必須先過濾 NaN 點再計算距離，
            # 否則 haversine 遇到 NaN 會回傳 NaN，np.sum 累加後整條距離變成 NaN
            # （導致 KPI 顯示 nan km、距離直方圖空白）。
            lon_v = d["lon"].values
            lat_v = d["lat"].values
            valid = np.isfinite(lon_v) & np.isfinite(lat_v)
            lon_v = lon_v[valid]
            lat_v = lat_v[valid]
            dist = np.sum(haversine(lon_v[:-1], lat_v[:-1], lon_v[1:], lat_v[1:])) if len(lon_v) >= 2 else 0.0
            dist_rows.append({
                "particle_id": pid, 
                "site_name": d["site_name"].iloc[0], 
                "distance_km": dist / 1000.0, 
                "final_status": d["status"].iloc[-1]
            })

        dist_df = pd.DataFrame(dist_rows)

        # 頂部關鍵群體指標
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("群體平均漂流里程", f"{dist_df['distance_km'].mean():.1f} km")
        kpi2.metric("最大漂流里程", f"{dist_df['distance_km'].max():.1f} km")
        
        beached_count = (dist_df["final_status"] == STATUS_LABELS[STATUS_BEACHED]).sum()
        kpi3.metric("沿岸陸源溯源率 (Terrestrial Origin)", f"{(beached_count / len(dist_df)) * 100:.1f}%")

        # 統計圖表雙欄呈現
        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            fig_hist = px.histogram(
                dist_df, x="distance_km", nbins=25, 
                title="漂流總里程分佈 (Travel Distance Distribution)",
                labels={"distance_km": "漂流距離 (km)", "count": "粒子數量"}
            )
            st.plotly_chart(fig_hist, use_container_width=True, config=PLOTLY_CONFIG,)
            # render_figure_export(fig_hist, "chart_travel_distance.png", label="漂流里程分佈圖", key="dl_hist")

        with col_fig2:
            fig_status = px.histogram(
                dist_df, x="final_status", 
                title="粒子最終來源歸宿分佈 (Particle Fate / Origin Distribution)",
                labels={"final_status": "回溯狀態類別", "count": "粒子數量"},
                color="final_status"
            )
            st.plotly_chart(fig_status, use_container_width=True, config=PLOTLY_CONFIG,)
            # render_figure_export(fig_status, "chart_particle_fate.png", label="粒子歸宿分佈圖", key="dl_status")

# ============================================================
# TAB 3 - 敏感度分析
# ============================================================
with tab3:
    if df_ready is None:
        st.info("請先執行模擬以進行敏感度分析。")
    else:
        st.subheader(
            "參數敏感度分析 (Parameter Sensitivity Analysis)",
            help=(
                "對單一參數掃描一組值，觀察「本地源佔比」隨參數變化的曲線。"
                "用以判斷結論的穩健性：若曲線跨越 50%，代表結論對該參數敏感、不穩健。"
            ),
        )

        # 敏感度分析參數設定
        sens_col1, sens_col2 = st.columns([1, 2])
        with sens_col1:
            sens_param = st.selectbox(
                "掃描參數 (Parameter to Sweep)",
                [
                    "海面風阻係數 (Windage %)",
                    "本地源關聯半徑 (Local Radius km)",
                    "回溯天數 (Days)",
                    "數值積分步長 (Time Step min)",
                ],
                key="sens_param",
            )
        with sens_col2:
            sens_n_points = st.slider(
                "掃描點數 (Number of Sweep Points)", min_value=3, max_value=9, value=5,
                key="sens_n_points",
                help=(
                    "在掃描範圍內要測試幾個參數值。系統會把掃描範圍平均切成 N 個點，"
                    "每個點都重跑一次完整模擬。點數越多，曲線越平滑、越能精確定位"
                    "「在哪個參數值會翻盤」，但計算時間也越長。"
                ),
            )

        # 依參數決定掃描範圍（以目前側邊欄設定為中心）。
        # 注意：windage 在內部為小數（0.015），但 UI 以百分比顯示，
        # 因此掃描範圍以百分比計算，傳入 run_sensitivity_sweep 前再除以 100。
        if sens_param.startswith("海面風阻"):
            _center = float(windage) * 100.0
            _lo = max(0.0, _center - 1.5)
            _hi = min(5.0, _center + 1.5)
            _param_key = "windage"
            _unit = "%"
        elif sens_param.startswith("本地源"):
            _center = float(local_source_radius_km)
            _lo = max(LOCAL_SOURCE_RADIUS_KM_MIN, _center - 10.0)
            _hi = min(LOCAL_SOURCE_RADIUS_KM_MAX, _center + 10.0)
            _param_key = "local_radius"
            _unit = " km"
        elif sens_param.startswith("回溯天數"):
            _center = float(_requested_days)
            _lo = max(1.0, _center - 3.0)
            _hi = min(30.0, _center + 3.0)
            _param_key = "days"
            _unit = " 天"
        else:
            _center = float(dt_mins)
            _lo = max(10.0, _center - 30.0)
            _hi = min(120.0, _center + 30.0)
            _param_key = "dt_mins"
            _unit = " min"

        _sweep_values = np.linspace(_lo, _hi, sens_n_points).tolist()
        # windage 掃描值以百分比計算，傳入模擬前轉回小數（與 ck.py 內部單位一致）。
        if _param_key == "windage":
            _sweep_values = [v / 100.0 for v in _sweep_values]
        st.caption(
            f"掃描範圍：{_lo:.1f}{_unit} ~ {_hi:.1f}{_unit}"
            f"（目前設定 {_center:.1f}{_unit}），共 {sens_n_points} 點。"
        )

        if st.button("執行敏感度掃描 (Run Sensitivity Sweep)", use_container_width=True):
            _sens_sites = (
                [(name, *HOTSPOT_CENTERS[name]) for name in HOTSPOT_CENTERS.keys()]
                if backtrack_all_hotspots else [(site_name, init_lon, init_lat)]
            )
            with st.spinner("正在執行參數掃描..."):
                sens_df = run_sensitivity_sweep(
                    param_name=_param_key,
                    param_values=_sweep_values,
                    base_kwargs={"windage": windage, "days": _requested_days, "dt_mins": dt_mins},
                    release_sites=_sens_sites,
                    total_particles=n_particles,
                    total_steps=total_steps,
                    dt_mins=dt_mins,
                    prov=providers,
                    reference_time_utc=reference_time_utc,
                    local_radius_km=local_source_radius_km,
                )
            st.session_state["sens_df"] = sens_df
            st.session_state["sens_param_label"] = sens_param
            st.session_state["sens_unit"] = _unit

        # 顯示掃描結果
        if "sens_df" in st.session_state:
            sens_df = st.session_state["sens_df"]
            _label = st.session_state.get("sens_param_label", sens_param)
            _u = st.session_state.get("sens_unit", _unit)

            # 穩健性判定：曲線是否跨越 50%
            _crosses = bool((sens_df["local_pct"].min() < 50.0) and (sens_df["local_pct"].max() > 50.0))
            _all_sig = bool(sens_df["significant"].all())
            if _crosses:
                st.error(
                    "⚠️ 結論不穩健 (Not Robust)：本地源佔比在掃描範圍內跨越 50%，"
                    "代表來源判定會隨此參數改變而翻盤。"
                )
            elif _all_sig:
                st.success(
                    "✅ 結論穩健 (Robust)：本地源佔比在掃描範圍內始終位於 50% 同一側，"
                    "且各點皆統計顯著。"
                )
            else:
                st.warning(
                    "⚠️ 結論部分穩健：本地源佔比未跨越 50%，但部分掃描點統計不顯著。"
                )

            # 敏感度曲線圖（含 95% CI 誤差帶）
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=sens_df["param_value"], y=sens_df["local_pct"],
                mode="lines+markers", name="本地源佔比",
                line=dict(color="#4C9AFF", width=3), marker=dict(size=9),
            ))
            fig_sens.add_trace(go.Scatter(
                x=pd.concat([sens_df["param_value"], sens_df["param_value"][::-1]]),
                y=pd.concat([sens_df["ci_hi"], sens_df["ci_lo"][::-1]]),
                fill="toself", fillcolor="rgba(76,154,255,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% 信賴區間", hoverinfo="skip",
            ))
            fig_sens.add_hline(
                y=50.0, line_dash="dash", line_color="#FF6B6B",
                annotation_text="50% 判定線", annotation_position="right",
            )
            fig_sens.update_layout(
                title=f"敏感度曲線：{_label}",
                xaxis_title=f"{_label}",
                yaxis_title="本地源佔比 (%)",
                yaxis_range=[0, 100],
                template="plotly_dark",
                height=420,
            )
            st.plotly_chart(fig_sens, use_container_width=True, config=PLOTLY_CONFIG,)
            # render_figure_export(fig_sens, "chart_sensitivity.png", label="敏感度曲線圖", key="dl_sens")

            # 掃描結果表格
            _tbl = sens_df.copy()
            _tbl["param_value"] = _tbl["param_value"].round(2)
            _tbl["local_pct"] = _tbl["local_pct"].round(1)
            _tbl["95% CI"] = _tbl.apply(lambda r: f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]", axis=1)
            _tbl["顯著"] = _tbl["significant"].map({True: "是", False: "否"})
            _tbl = _tbl.rename(columns={
                "param_value": f"參數值 ({_u.strip()})",
                "local_pct": "本地源佔比 (%)",
                "net_displacement_km": "淨位移 (km)",
                "prediction": "來源推論傾向",
            })
            st.dataframe(
                _tbl[[f"參數值 ({_u.strip()})", "本地源佔比 (%)", "95% CI", "顯著",
                      "淨位移 (km)", "來源推論傾向"]],
                use_container_width=True, height=240,
            )