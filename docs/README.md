# ETF Fund Portfolio Investment Analysis System

一个面向场外 ETF 联接基金场景的 Python 分析系统，覆盖数据获取与清洗、指标计算、选基、策略、交易模拟、回测、报告生成，以及 Flask Web 演示界面。

## 1. 项目目标与当前形态

- 目标：围绕 A 股 ETF 投资流程，打通 `数据 -> 公式 -> 策略 -> 模拟 -> 回测 -> 报告/页面`。
- 当前代码形态：核心模块已落地，测试覆盖较完整（本地 `pytest -q` 为 176 通过）。
- Web 层定位：用于演示和交互验证，后端接口在 `src/etf_analyzer/web/app.py`。

## 2. 目录结构（开发入口）

```text
ETF/
├── src/etf_analyzer/
│   ├── core/          # 配置、日志、统一响应、交易日历、缓存
│   ├── data/          # akshare 抓取、清洗、存储、增量更新
│   ├── formulas/      # 收益/估值/风险/技术/因子公式
│   ├── selection/     # 筛选与评分
│   ├── strategy/      # 策略基类、信号定义、半月调仓策略
│   ├── simulation/    # 费用、券商、持仓、定投模拟
│   ├── backtest/      # 回测引擎、指标聚合、可视化、HTML 报告
│   └── web/           # Flask 工厂与页面/API 路由
├── tests/             # 对应模块测试 + 集成测试
├── config/            # settings/data_sources/strategy_params
├── templates/         # report.html + web 页面模板
├── static/            # web 静态资源
├── pyproject.toml
└── requirements.txt
```

## 3. 环境准备

### 3.1 Python 版本

- 要求：Python `>= 3.11`（见 `pyproject.toml`）。

### 3.2 创建虚拟环境并安装依赖

```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS/Linux
# python -m venv .venv
# source .venv/bin/activate

# 推荐：用 requirements.txt（包含 flask）
pip install -r requirements.txt

# 可选：开发安装
pip install -e ".[dev]"
```

说明：`requirements.txt` 包含 `flask`，而 `pyproject.toml` 的 dependencies 当前未列出 `flask`。如果你仅执行 `pip install -e ".[dev]"`，请确认 Flask 已安装。

## 4. 本地运行

### 4.1 运行测试

```bash
pytest -q
```

### 4.2 启动 Web 应用

项目使用 app factory（`create_app`）：

```bash
flask --app etf_analyzer.web.app:create_app run --debug --port 5000
```

启动后访问：

- 首页：`http://127.0.0.1:5000/`
- 回测页：`/backtest`
- 筛选页：`/screening`
- 策略页：`/strategy`
- 数据页：`/data`
- 定投页：`/dca`

## 5. 核心开发流程（给下一个 AI/开发者）

建议按以下顺序推进，和现有代码结构保持一致。

1. 明确需求落点（模块）
   - 数据问题优先看 `data/`
   - 信号逻辑优先看 `strategy/`
   - 交易与费用优先看 `simulation/`
   - 指标、回测结果优先看 `backtest/` + `formulas/`
2. 先补/改测试，再改实现
   - 测试目录与源码目录 1:1 对齐，优先写对应 `tests/test_xxx/`。
3. 保持统一接口风格
   - 模块间统一用 `ApiResponse`（`core/response.py`）承载 `status_code/data/message`。
4. 参数配置优先
   - 策略和系统常量优先放 `config/*.yaml`，避免硬编码。
5. 完成后验证
   - 至少跑目标测试 + 全量 `pytest -q`。

## 6. 常见任务指引

### 6.1 更新 ETF 数据

典型流程：`EtfDataFetcher -> DataUpdater -> EtfDataStore`。

参考实现位置：

- `src/etf_analyzer/data/fetcher.py`
- `src/etf_analyzer/data/updater.py`
- `src/etf_analyzer/data/store.py`

### 6.2 跑策略回测

核心入口：

- `BacktestEngine`：`src/etf_analyzer/backtest/engine.py`
- `calculate_backtest_metrics`：`src/etf_analyzer/backtest/metrics.py`
- 集成样例：`tests/test_integration.py`

### 6.3 生成 HTML 报告

- 报告生成函数：`generate_html_report` in `src/etf_analyzer/backtest/report.py`
- 默认模板：`templates/report.html`

### 6.4 定投模拟

- 入口：`DCASimulator` in `src/etf_analyzer/simulation/dca.py`
- 相关 API：`/api/dca/run`（`src/etf_analyzer/web/app.py`）

## 7. Web API 速览

主要接口在 `src/etf_analyzer/web/app.py`：

- `GET /api/health` 健康检查
- `POST /api/backtest/run` 执行回测（当前为演示数据）
- `POST /api/backtest/validate` 样本内外验证
- `POST /api/backtest/sensitivity` 参数敏感性分析
- `POST /api/dca/run` 定投模拟
- `GET /api/data/etf/<code>` ETF 数据查询
- `POST /api/selection/screen` 选基筛选
- `GET/PUT /api/strategy/params` 策略参数读取/更新
- `POST /api/strategy/signals` 策略信号预览

## 8. 部署建议（用户使用）

### 8.1 开发环境

- 使用 Flask 内置 server（仅开发调试）：
  - `flask --app etf_analyzer.web.app:create_app run --debug`

### 8.2 生产环境

- Linux 推荐 Gunicorn：

```bash
gunicorn -w 4 -b 0.0.0.0:8000 'etf_analyzer.web.app:create_app()'
```

- Windows 推荐 Waitress：

```bash
waitress-serve --listen=*:8000 etf_analyzer.web.app:create_app
```

实践建议：

- 不要在生产开启 debug。
- 通过环境变量管理敏感配置。
- 用反向代理（如 Nginx）处理静态资源和 HTTPS。

## 9. 已知注意点

- 依赖声明存在轻微分散：`flask` 在 `requirements.txt`，不在 `pyproject.toml` 主依赖中。
- 设计文档为 `概要设计文档.md`，实现可能按工程现实做过收敛（以代码和测试为准）。

## 10. 给 AI 的快速执行清单

每次开始改动前，按这个顺序做：

1. 先读 `pyproject.toml`、`config/*.yaml`、目标模块与同目录测试。
2. 用测试定义改动边界：先写/改失败测试，再改实现。
3. 改完至少执行：
   - `pytest tests/<目标模块> -q`
   - `pytest -q`
4. 变更涉及 Web 接口时，补充/更新 `tests/test_web/test_app.py`。
5. 若新增配置项，必须同步更新 `config/*.yaml` 与本 Readme 的对应章节。
