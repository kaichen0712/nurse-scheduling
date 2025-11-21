# core/nurse_scheduling/scheduler.py
import itertools
import logging
import time
from datetime import timedelta
from typing import Callable, Optional

from ortools.sat.python import cp_model

from . import exporter, preference_types
from .context import Context
from .utils import ortools_expression_to_bool_var, parse_dates, MAP_DATE_KEYWORD_TO_FILTER, MAP_WEEKDAY_TO_STR
from .constants import ALL, OFF, OFF_sid
from .loader import load_data


# 舊函數保留相容性
def schedule(file_content: bytes, deterministic=False, avoid_solution=None, prettify=False, timeout: int | None = None):
    return schedule_with_logger(
        file_content=file_content,
        deterministic=deterministic,
        avoid_solution=avoid_solution,
        prettify=prettify,
        timeout=timeout,
        logger=None,  # 不傳 logger 
        testing=True,
    )


# 全新主函數：支援即時 logger callback
def schedule_with_logger(
    file_content: bytes,
    deterministic=False,
    avoid_solution=None,
    prettify=False,
    timeout: int | None = None,
    logger: Optional[Callable[[str], None]] = None,
    testing: bool = False,  # 👉 新增：測試模式開關
):
    """
    logger: 如果傳入，就會即時呼叫 logger("訊息\n")，用於 SSE
            如果是 None，就退回原本的 logging.info
    """
    def log(msg: str):
        if logger:
            logger(msg + "\n")
        else:
            logging.info(msg)

    log("載入排班設定檔...")
    scenario = load_data(file_content)

    log("解析設定資料...")
    if scenario.apiVersion != "alpha":
        raise NotImplementedError(f"不支援的 API 版本: {scenario.apiVersion}")

    ctx = Context(**dict(scenario))
    del scenario
    ctx.n_days = (ctx.dates.range.endDate - ctx.dates.range.startDate).days + 1
    ctx.n_shift_types = len(ctx.shiftTypes.items)
    ctx.n_people = len(ctx.people.items)
    ctx.dates.items = [ctx.dates.range.startDate + timedelta(days=d) for d in range(ctx.n_days)]

    log(f"排班期間：{ctx.n_days} 天 | 班別數：{ctx.n_shift_types} | 人員數：{ctx.n_people}")

 
    # Map shift type ID to shift type index
    for s in range(ctx.n_shift_types):
        ctx.map_sid_s[ctx.shiftTypes.items[s].id] = [s]
    ctx.map_sid_s[ALL] = list(range(ctx.n_shift_types))
    ctx.map_sid_s[OFF] = [OFF_sid]
    for g in range(len(ctx.shiftTypes.groups)):
        group = ctx.shiftTypes.groups[g]
        ctx.map_sid_s[group.id] = sorted(set().union(*[ctx.map_sid_s[sid] for sid in group.members]))

    for p in range(ctx.n_people):
        ctx.map_pid_p[ctx.people.items[p].id] = [p]
    ctx.map_pid_p[ALL] = list(range(ctx.n_people))
    for g in range(len(ctx.people.groups)):
        group = ctx.people.groups[g]
        ctx.map_pid_p[group.id] = sorted(set().union(*[ctx.map_pid_p[pid] for pid in group.members]))

    for d in range(ctx.n_days):
        date_obj = ctx.dates.items[d]
        ctx.map_did_d[str(date_obj)] = [d]
    for keyword in MAP_DATE_KEYWORD_TO_FILTER:
        ctx.map_did_d[keyword] = [d for d in range(ctx.n_days) if MAP_DATE_KEYWORD_TO_FILTER[keyword](ctx.dates.items[d])]
    for keyword in MAP_WEEKDAY_TO_STR:
        weekday_index = MAP_WEEKDAY_TO_STR.index(keyword)
        ctx.map_did_d[keyword] = [d for d in range(ctx.n_days) if ctx.dates.items[d].weekday() == weekday_index]
    for g in range(len(ctx.dates.groups)):
        group = ctx.dates.groups[g]
        date_indices = set()
        for member in group.members:
            if member in ctx.map_did_d:
                date_indices.update(ctx.map_did_d[member])
            else:
                date_indices.update(parse_dates(member, ctx.map_did_d, ctx.dates.range))
        ctx.map_did_d[group.id] = sorted(set(date_indices))

    log("建立排班變數 (d, s, p)...")
    for d in range(ctx.n_days):
        for s in range(ctx.n_shift_types):
            for p in range(ctx.n_people):
                var_name = f"shift_d{d}_s{s}_p{p}"
                ctx.model_vars[var_name] = ctx.shifts[(d, s, p)] = ctx.model.NewBoolVar(var_name)

    if avoid_solution is not None:
        log("避開先前解...")
        avoid_solution_vars = []
        for (d, s, p) in ctx.shifts:
            val = avoid_solution.get((d, s, p), 0)
            if val == 0:
                avoid_solution_vars.append(ctx.shifts[(d, s, p)])
            elif val == 1:
                avoid_solution_vars.append(ctx.shifts[(d, s, p)].Not())
        ctx.model.AddBoolOr(avoid_solution_vars)

    log("建立休假變數...")
    for d in range(ctx.n_days):
        for p in range(ctx.n_people):
            dp_shifts_sum = sum(ctx.shifts[(d, s, p)] for s in range(ctx.n_shift_types))
            var_name = f"off_d{d}_p{p}"
            ctx.model_vars[var_name] = ctx.offs[(d, p)] = ortools_expression_to_bool_var(
                ctx.model, var_name, dp_shifts_sum == 0, dp_shifts_sum != 0
            )

    log("建立快速查詢索引...")
    ctx.map_ds_p = {(d, s): {p for p in range(ctx.n_people) if (d, s, p) in ctx.shifts} for (d, s) in itertools.product(range(ctx.n_days), range(ctx.n_shift_types))}
    ctx.map_dp_s = {(d, p): {s for s in range(ctx.n_shift_types) if (d, s, p) in ctx.shifts} for (d, p) in itertools.product(range(ctx.n_days), range(ctx.n_people))}
    ctx.map_d_sp = {d: {(s, p) for (s, p) in itertools.product(range(ctx.n_shift_types), range(ctx.n_people)) if (d, s, p) in ctx.shifts} for d in range(ctx.n_days)}
    ctx.map_s_dp = {s: {(d, p) for (d, p) in itertools.product(range(ctx.n_days), range(ctx.n_people)) if (d, s, p) in ctx.shifts} for s in range(ctx.n_shift_types)}
    ctx.map_p_ds = {p: {(d, s) for (d, s) in itertools.product(range(ctx.n_days), range(ctx.n_shift_types)) if (d, s, p) in ctx.shifts} for p in range(ctx.n_people)}

    log("載入所有偏好與約束條件...")
    for i, preference in enumerate(ctx.preferences):
        preference_types.PREFERENCE_TYPES_TO_FUNC[preference.type](ctx, preference, i)

    ctx.model.Maximize(ctx.objective)

    log("初始化 CP-SAT 求解器...")
    solver = cp_model.CpSolver()
    if deterministic:
        solver.parameters.random_seed = 0
        solver.parameters.num_workers = 1

    # 即時顯示每一個更好解
    class LiveSolutionPrinter(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.solution_count = 0
            self.start_time = time.time()

        def on_solution_callback(self):
            self.solution_count += 1
            current = self.Value(ctx.objective)
            elapsed = time.time() - self.start_time
            log(f"找到更好解！第 {self.solution_count} 個 | 分數 = {current} | 耗時 {elapsed:.1f}s")

    solution_printer = LiveSolutionPrinter()

    # 讓 OR-Tools 自己印的 log 推到前端
    solver.parameters.log_search_progress = not testing  # 測試模式下關閉，避免干擾測試輸出

    if timeout is not None:
        solver.parameters.max_time_in_seconds = float(timeout)
        log(f"時間限制：{timeout} 秒")

    log("開始求解！（即時日誌如下）")
    if testing:
        status = solver.Solve(ctx.model)
    else:
        solution_printer = LiveSolutionPrinter()
        status = solver.Solve(ctx.model, solution_printer)

    log(f"求解結束！狀態：{solver.StatusName(status)}")

    found = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if status == cp_model.OPTIMAL:
        log("找到最佳解！")
    elif status == cp_model.FEASIBLE:
        log("找到可行解！")
    else:
        log("無解或超時")

    if not found:
        return None, None, None, solver.StatusName(status), None

    score = solver.Value(ctx.objective)
    log(f"最終分數：{score}")

    df, cell_export_info = exporter.get_people_versus_date_dataframe(ctx, solver, prettify=prettify)

    solution = {(d, s, p): solver.Value(ctx.shifts[(d, s, p)]) for (d, s, p) in ctx.shifts}

    return df, solution, score, solver.StatusName(status), cell_export_info