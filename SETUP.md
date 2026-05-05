# Windows 環境安裝與啟動指南

本文件說明如何在 Windows 環境下從零開始安裝並啟動護理排班系統的後端與前端。

---

## 事前準備

### 1. 安裝 uv（Python 套件管理工具）

開啟 **PowerShell**，執行以下指令：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安裝完成後，**關閉並重新開啟 PowerShell**，讓 PATH 生效。

確認安裝成功：

```powershell
uv --version
```

### 2. 安裝 Node.js（前端用）

前往 [https://nodejs.org/](https://nodejs.org/) 下載並安裝 LTS 版本。

確認安裝成功：

```powershell
node --version
npm --version
```

---

## 取得專案

```powershell
git clone https://github.com/j3soon/nurse-scheduling.git
cd nurse-scheduling
```

---

## 後端（Core + Web Backend）

### 步驟一：建立 Python 虛擬環境

```powershell
cd core
uv venv --python 3.12
```

> `uv` 會自動下載 Python 3.12，不需要另外安裝。

### 步驟二：啟動虛擬環境

```powershell
.venv\Scripts\activate
```

啟動成功後，命令提示字元前方會出現 `(.venv)` 標示。

### 步驟三：安裝依賴套件

```powershell
uv pip install -r requirements.txt
```

### 步驟四：啟動後端伺服器（開發模式）

```powershell
cd nurse_scheduling
fastapi dev serve.py
```

後端預設執行於 `http://127.0.0.1:8000`，支援熱重載。

### 離開虛擬環境

```powershell
deactivate
```

---

## 前端（Web Frontend）

開啟**另一個** PowerShell 視窗：

```powershell
cd web-frontend
npm install
npm run dev
```

前端預設執行於 `http://localhost:3000`。

---

## 日常啟動流程（已安裝完成後）

每次要啟動系統時，需要同時開啟兩個終端機：

**終端機 1 — 後端：**

```powershell
cd D:\path\to\nurse-scheduling\core
.venv\Scripts\activate
cd nurse_scheduling
fastapi dev serve.py
```

> 注意：`.venv` 在 `core` 資料夾內，需先進入 `core` 才能啟動虛擬環境。

**終端機 2 — 前端：**

```powershell
cd D:\path\to\nurse-scheduling\web-frontend
npm run dev
```

---

## 更新依賴套件

若 `requirements.txt` 有變動，重新安裝：

```powershell
cd core
.venv\Scripts\activate
uv pip install -r requirements.txt
```

---

## 常見問題

### `uv: command not found`

重新開啟 PowerShell 或手動將 `uv` 加入 PATH：

```powershell
$env:Path = "C:\Users\你的帳號\.local\bin;$env:Path"
```

### 排班時出現「Date is outside valid range」錯誤

需要在 [core/nurse_scheduling/workdays/taiwan.py](core/nurse_scheduling/workdays/taiwan.py) 新增對應年份的台灣國定假日資料，並更新 `valid_date_range`。
