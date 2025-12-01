import ipywidgets as widgets
from IPython.display import display, clear_output
import datetime
from zoneinfo import ZoneInfo
import math
import numpy as np

def fermentation_speed_factor(temp_F, has_culture, Q10=2.0, culture_boost=1.5):
    T_ref_C = (68.0 - 32) * 5/9
    T_user_C = (temp_F - 32) * 5/9
    temp_factor = Q10 ** ((T_user_C - T_ref_C) / 10.0)
    cult_factor = culture_boost if has_culture else 1.0
    return temp_factor * cult_factor

def format_delta_hours(delta_h):
    d = abs(delta_h)
    days = int(d // 24)
    hours = int(round(d % 24))
    if days == 0 and hours == 0:
        return "≈ now"
    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    return "in ~" + " and ".join(parts)

def stage_from_hours(hours_equiv_68, has_culture_bool):
    ph = float(reg_time.predict([[hours_equiv_68, int(has_culture_bool)]])[0])
    days = hours_equiv_68 / 24.0
    st = heuristic_stage(ph, days)
    if has_culture_bool and st.lower() == "kefir":
        st = "yogurt"
    if (not has_culture_bool) and st.lower() == "yogurt":
        st = "kefir"
    return st, ph

def model_future_times(hours_now, has_culture_bool, step_hours=0.5, extra_days=10.0):
    horizon = max(MAX_HOURS_DATA, hours_now) + extra_days * 24.0
    times = np.arange(hours_now, horizon + step_hours / 2.0, step_hours)
    has_vec = np.full_like(times, int(has_culture_bool), dtype=float)
    X = np.column_stack([times, has_vec])
    ph_seq = reg_time.predict(X)

    stages = []
    for t, ph in zip(times, ph_seq):
        days = t / 24.0
        st = heuristic_stage(ph, days)
        if has_culture_bool and st.lower() == "kefir":
            st = "yogurt"
        if (not has_culture_bool) and st.lower() == "yogurt":
            st = "kefir"
        stages.append(st)

    current_stage = stages[0]
    first_model = {}
    for st in STAGE_ORDER:
        if has_culture_bool and st.lower() == "kefir":
            continue
        if (not has_culture_bool) and st.lower() == "yogurt":
            continue
        if st == current_stage:
            continue
        idx = np.where(np.array(stages) == st)[0]
        if len(idx) > 0:
            first_model[st] = float(times[idx[0]])

    return current_stage, first_model

title = widgets.HTML(
    "<b>Interactive raw-milk fermentation assistant "
    "(fresh → sour → yogurt/kefir → curding → cheese)</b>"
)

now_et = datetime.datetime.now(ZoneInfo("America/New_York"))

month_dropdown = widgets.Dropdown(
    options=list(zip(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        range(1,13)
    )),
    value=now_et.month,
    layout=widgets.Layout(width="90px")
)
day_dropdown = widgets.Dropdown(
    options=list(range(1,32)),
    value=now_et.day,
    layout=widgets.Layout(width="70px")
)
year_dropdown = widgets.Dropdown(
    options=[now_et.year],
    value=now_et.year,
    layout=widgets.Layout(width="80px")
)

hour_dropdown = widgets.Dropdown(
    options=list(range(1,13)),
    value=((now_et.hour - 1) % 12) + 1,
    layout=widgets.Layout(width="60px")
)
minute_dropdown = widgets.Dropdown(
    options=[0, 30],
    value=0 if now_et.minute < 30 else 30,
    layout=widgets.Layout(width="60px")
)
ampm_dropdown = widgets.Dropdown(
    options=["AM", "PM"],
    value="PM" if now_et.hour >= 12 else "AM",
    layout=widgets.Layout(width="70px")
)

temp_input = widgets.BoundedFloatText(
    value=68.0,
    min=50.0,
    max=90.0,
    step=0.5,
    layout=widgets.Layout(width="80px")
)

culture_dropdown = widgets.Dropdown(
    options=[("no", False), ("yes", True)],
    value=False,
    layout=widgets.Layout(width="70px")
)

predict_button = widgets.Button(
    description="Predict",
    button_style="success",
    layout=widgets.Layout(width="80px")
)

status_label = widgets.Label(
    "Choose start date/time, temp, culture, then click 'Predict'."
)

output = widgets.Output()

controls_row = widgets.HBox([
    widgets.Label("Start:", layout=widgets.Layout(width="45px")),
    month_dropdown,
    day_dropdown,
    year_dropdown,
    widgets.Label("Time:", layout=widgets.Layout(width="40px")),
    hour_dropdown,
    widgets.Label(":", layout=widgets.Layout(width="10px")),
    minute_dropdown,
    ampm_dropdown,
    widgets.Label("Temp°F:", layout=widgets.Layout(width="60px")),
    temp_input,
    widgets.Label("Culture:", layout=widgets.Layout(width="60px")),
    culture_dropdown,
    predict_button
])

def on_predict_clicked(b):
    with output:
        clear_output()

        year = year_dropdown.value
        month = month_dropdown.value
        day = day_dropdown.value
        h12 = hour_dropdown.value
        minute = int(minute_dropdown.value)
        ampm = ampm_dropdown.value

        if ampm == "PM" and h12 != 12:
            h24 = h12 + 12
        elif ampm == "AM" and h12 == 12:
            h24 = 0
        else:
            h24 = h12

        try:
            start_dt = datetime.datetime(
                year, month, day, h24, minute, tzinfo=ZoneInfo("America/New_York")
            )
        except ValueError as e:
            status_label.value = f"Invalid date: {e}"
            return

        now = datetime.datetime.now(ZoneInfo("America/New_York"))
        elapsed_h = (now - start_dt).total_seconds() / 3600.0
        if elapsed_h < 0:
            status_label.value = "Start time is in the future — choose an earlier date/time."
            return

        temp_F = float(temp_input.value)
        has_culture = culture_dropdown.value

        speed_temp_only = fermentation_speed_factor(temp_F, False)
        hours_equiv_68 = elapsed_h * speed_temp_only
        days_equiv_68 = hours_equiv_68 / 24.0

        stage_now, ph_est = stage_from_hours(hours_equiv_68, has_culture)

        status_label.value = (
            f"Batch started {start_dt.strftime('%Y-%m-%d %I:%M %p')} ET, "
            f"{elapsed_h:.1f}h elapsed at {temp_F:.1f}°F "
            f"({'culture' if has_culture else 'no culture'})."
        )

        print("=== Fermentation Status (model + labels) ===")
        print(f"Start (ET):  {start_dt.strftime('%Y-%m-%d %I:%M %p')}")
        print(f"Now (ET):    {now.strftime('%Y-%m-%d %I:%M %p')}")
        print(f"Elapsed real time: {elapsed_h:.1f} h")
        print(f"Temperature: {temp_F:.1f}°F")
        print(f"Culture added: {'Yes' if has_culture else 'No'}")
        print(f"Effective 68°F hours (temp only): {hours_equiv_68:.1f} h "
              f"({days_equiv_68:.1f} days at 68°F)\n")

        print(f"Estimated pH (reg_time): ~{ph_est:.2f}")
        print(f"Estimated stage: {stage_now}\n")

        print("=== Upcoming Stage Changes (same model, with label fallback) ===")

        model_current, model_first = model_future_times(hours_equiv_68, has_culture)

        if has_culture:
            label_first = stage_first_times_cultured
        else:
            label_first = stage_first_times_noculture

        if stage_now in STAGE_ORDER:
            current_idx = STAGE_ORDER.index(stage_now)
        else:
            current_idx = -1

        future_map = {}
        for st in STAGE_ORDER:
            if has_culture and st.lower() == "kefir":
                continue
            if (not has_culture) and st.lower() == "yogurt":
                continue
            if st not in STAGE_ORDER:
                continue
            idx = STAGE_ORDER.index(st)
            if idx <= current_idx:
                continue

            candidates = []

            t_model = model_first.get(st)
            if t_model is not None and t_model > hours_equiv_68:
                candidates.append(t_model)

            t_label = label_first.get(st)
            if t_label is not None and t_label > hours_equiv_68:
                candidates.append(t_label)

            if candidates:
                t_best = min(candidates)

                if (not has_culture) and (st.lower() == "curding"):
                    min_cur_hours = 35.0 * 24.0
                    if t_best < min_cur_hours:
                        t_best = min_cur_hours

                future_map[st] = t_best

        if not future_map:
            print("All remaining stages have already passed according to this model.")
        else:
            for st in STAGE_ORDER:
                if st not in future_map:
                    continue
                delta = future_map[st] - hours_equiv_68
                when = format_delta_hours(delta)
                print(f"  {st:15s}: {when}")

predict_button.on_click(on_predict_clicked)

ui = widgets.VBox([
    title,
    controls_row,
    status_label,
    output
])

display(ui)
