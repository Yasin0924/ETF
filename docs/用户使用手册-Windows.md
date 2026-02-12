# ETF 智能投研系统用户使用手册（Windows）

## 1. 适用对象

本手册面向在 Windows 系统上本地部署和使用本系统的用户。

## 2. 环境要求

- Windows 10/11
- Python 3.11+
- 可访问互联网（用于首次拉取真实数据）

## 3. 安装步骤

在 PowerShell 中进入项目目录后执行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果 PowerShell 禁止执行脚本，可先执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 4. 启动系统

```powershell
flask --app etf_analyzer.web.app:create_app run --debug --port 5000
```

打开浏览器访问：`http://127.0.0.1:5000`

主要页面：

- 组合总览：`/`
- 策略回测：`/backtest`
- 基金筛选：`/screening`
- 策略参数：`/strategy`
- 定投模拟：`/dca`
- 数据查询：`/data`

## 5. 首次使用建议流程

1. 先到“数据查询”页查询 510300（验证真实数据链路）
2. 再到“基金筛选”页执行筛选
3. 若有结果，可“加入策略池”并到“策略参数”页一键应用
4. 到“策略回测”页执行回测、样本外验证、敏感性分析
5. 到“定投模拟”页对同类标的做长期定投对比

## 6. 页面功能说明

### 6.1 组合总览

- 显示回测核心指标、净值曲线、最近交易日志
- 显示风控状态与系统状态（数据源、缓存、校验）
- 支持“刷新数据”

### 6.2 基金筛选

- 可按规模、费率、分类、估值分位、ROE、跟踪误差筛选
- 支持结果排序与导出
- 无结果时会显示推荐参数，可一键应用并重试

### 6.3 策略参数

- 支持在线编辑参数并即时更新
- 带参数校验提示
- 可读取候选基金池并一键应用为目标权重

### 6.4 策略回测

- 执行回测并展示关键指标和交易日志
- 支持样本外验证
- 支持参数敏感性分析
- 支持导出回测结果

### 6.5 定投模拟

- 支持固定金额/固定份额
- 支持现金分红/红利再投资入口
- 显示分笔记录（计划日期、成交日期、费用、份额）

### 6.6 数据查询

- 查询指定 ETF 在日期区间内的真实数据
- 展示数据来源、缓存状态与更新时间

## 7. 真实数据说明

- 系统优先使用本地历史数据（`data/etf/*.csv`）
- 本地不足时自动尝试实时拉取并更新
- 前端会显示数据来源：
  - 实时数据
  - 本地历史数据
  - 模拟数据（仅特定场景）

## 8. 常见问题（Windows）

### 8.1 启动后页面打不开

- 检查端口是否被占用
- 改用其他端口：`--port 5001`

### 8.2 抓取真实数据失败

可能原因：代理配置、网络波动、数据源临时不可用。

排查建议：

```powershell
gci env: | ? { $_.Name -match 'proxy' }
nslookup push2his.eastmoney.com
Test-NetConnection push2his.eastmoney.com -Port 443
```

如果当前终端需要临时直连：

```powershell
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
```

### 8.3 筛选结果为空

这不一定是错误，通常是条件过严。可直接使用页面给出的推荐参数并一键重试。

### 8.4 页面内容更新不及时

- 浏览器强制刷新：`Ctrl + F5`
- 确保 Flask 服务已重启

## 9. 日常维护建议

- 定期执行一次数据查询/回测，触发本地数据更新
- 关注日志中的 ERROR/WARNING（特别是数据源连接问题）
- 升级依赖前先运行全量测试

## 10. 升级与回归检查

每次升级后建议执行：

```powershell
pytest tests/test_web/test_app.py -q
pytest -q
```

若测试全部通过，再继续日常使用。
