# zTSP_1205/test_gls_engine_perturb_grid.py
import time
import numpy as np

from gls.spec import default_gls_spec
from prob import TSPGLS


class BuiltinGLS:
    """
    3.2: 内置版 GLS 惩罚策略（不依赖 LLM），接口与 LLM 版保持一致：
      updated_edge_distance = update_edge_distance(edge_distance, local_opt_tour, edge_n_used)
    """
    lam = 0.5  # 默认惩罚强度，可在脚本中通过 BuiltinGLS.lam 调整

    @staticmethod
    def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
        lam = float(BuiltinGLS.lam)
        return edge_distance + lam * edge_n_used


def configure_engine(spec, engine_type: str, engine_params: dict | None = None):
    """
    根据 engine_type 和自定义参数，填充 spec.engine。
    - 对 ls_basic：目前只需要设置 type；
    - 对 ls_lk / ls_lk_tail：使用 GLSSpec.engine 中的 LKH-lite / ILS 参数，并可被 engine_params 覆盖。
    """
    spec.engine["type"] = engine_type

    if engine_type in ("ls_lk", "ls_lk_tail"):
        # 先确保有默认值
        spec.engine.setdefault("lk_max_outer", 6)
        spec.engine.setdefault("lk_max_inner", 2)
        spec.engine.setdefault("lk_top_k", 20)
        spec.engine.setdefault("lk_restarts", 3)
        spec.engine.setdefault("lk_seed", 2025)
        spec.engine.setdefault("lk_first_improvement", False)
        spec.engine.setdefault("lk_tail_reserve_s", 2.0)

        if engine_params is not None:
            for k, v in engine_params.items():
                spec.engine[k] = v


def run_setting(
    prob: TSPGLS,
    name: str,
    heuristic,
    engine_type: str,
    perturb_type: str,
    k_list,
    lam_list,
    time_limit_s: float,
    multi_start: int = 1,
    engine_params: dict | None = None,
):
    print(f"\n=== [{name}] engine={engine_type}, perturb={perturb_type} ===")

    best_gap = float("inf")
    best_cfg = None

    for k in k_list:
        for lam in lam_list:
            spec = default_gls_spec()

            # 时间与迭代配置
            spec.stopping["time_limit_s"] = time_limit_s
            spec.schedule["loop_max"] = 400
            spec.schedule["max_no_improve"] = 80

            # 候选集 / 初始化
            spec.candset["k"] = int(k)
            spec.init["multi_start"] = int(multi_start)

            # 扰动配置
            spec.perturb["type"] = perturb_type
            if perturb_type != "none":
                spec.perturb["moves"] = 1
                spec.perturb["interval"] = 80

            # 引擎配置
            configure_engine(spec, engine_type, engine_params)

            # guidance 配置
            if heuristic is BuiltinGLS:
                spec.guidance["type"] = "builtin"
                BuiltinGLS.lam = float(lam)
                heuristic_module = BuiltinGLS
            else:
                spec.guidance["type"] = "none"
                heuristic_module = None

            t0 = time.time()
            gap = prob.evaluateGLS_with_spec(heuristic_module, spec)
            elapsed = time.time() - t0

            if gap >= 1e9:
                print(
                    f"[{name}] k={k}, lam={lam:.2f} "
                    f"| gap=INF | time={elapsed:.3f}s (error)"
                )
                continue

            print(
                f"[{name}] k={k}, lam={lam:.2f} "
                f"| gap={gap:.4f}% | time={elapsed:.3f}s"
            )

            if gap < best_gap:
                best_gap = gap
                best_cfg = (k, lam, elapsed)

    print(f"\n>>> [{name}] Best config")
    if best_cfg is not None:
        k, lam, elapsed = best_cfg
        print(
            f"    k={k}, lam={lam:.2f} "
            f"| best_gap={best_gap:.4f}% | time={elapsed:.3f}s"
        )
    else:
        print("    No valid run.")


def main():
    prob = TSPGLS()

    time_limit_s = 10.0
    k_list = [35, 40, 45, 50]
    lam_list_for_gls = [0.40, 0.50, 0.60]
    lam_list_dummy = [0.0]
    multi_start = 1

    # LKH-lite 引擎参数（可再微调）
    lk_engine_params = {
        "lk_max_outer": 6,
        "lk_max_inner": 2,
        "lk_top_k": 20,
        "lk_restarts": 3,
        "lk_seed": 2025,
        "lk_first_improvement": False,
        # 主循环预留给尾部精修的时间（秒）
        "lk_tail_reserve_s": 3.0,
    }

    run_setting(
        prob=prob,
        name="3.1_basic_none",
        heuristic=None,
        engine_type="ls_basic",
        perturb_type="none",
        k_list=k_list,
        lam_list=lam_list_dummy,
        time_limit_s=time_limit_s,
        multi_start=multi_start,
    )

    run_setting(
        prob=prob,
        name="3.1_basic_reloc",
        heuristic=None,
        engine_type="ls_basic",
        perturb_type="random_relocate",
        k_list=k_list,
        lam_list=lam_list_dummy,
        time_limit_s=time_limit_s,
        multi_start=multi_start,
    )

    run_setting(
        prob=prob,
        name="3.2_basic_none",
        heuristic=BuiltinGLS,
        engine_type="ls_basic",
        perturb_type="none",
        k_list=k_list,
        lam_list=lam_list_for_gls,
        time_limit_s=time_limit_s,
        multi_start=multi_start,
    )

    run_setting(
        prob=prob,
        name="3.2_basic_reloc",
        heuristic=BuiltinGLS,
        engine_type="ls_basic",
        perturb_type="random_relocate",
        k_list=k_list,
        lam_list=lam_list_for_gls,
        time_limit_s=time_limit_s,
        multi_start=multi_start,
    )

    # 组合 5：3.2_builtin_gls + GLS+尾部 LKH-lite 精修 + 无扰动
    run_setting(
        prob=prob,
        name="3.2_lk_tail_none",
        heuristic=BuiltinGLS,
        engine_type="ls_deep",
        perturb_type="none",
        k_list=k_list,
        lam_list=lam_list_for_gls,
        time_limit_s=time_limit_s,
        multi_start=multi_start,
        engine_params=lk_engine_params,
    )

    # 组合 6：3.2_builtin_gls + GLS+尾部 LKH-lite 精修 + random_relocate 扰动
    run_setting(
        prob=prob,
        name="3.2_lk_tail_reloc",
        heuristic=BuiltinGLS,
        engine_type="ls_deep",
        perturb_type="random_relocate",
        k_list=k_list,
        lam_list=lam_list_for_gls,
        time_limit_s=time_limit_s,
        multi_start=multi_start,
        engine_params=lk_engine_params,
    )


if __name__ == "__main__":
    main()
