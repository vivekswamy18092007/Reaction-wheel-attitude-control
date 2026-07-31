# DEIMoS Demo Video Script — Simulation Part (target 2:30)

Your segment, following Tulika's 2:00 CAD segment.

---

### 0:00–0:10 — Intro

**ON SCREEN:** Cut from Tulika's CAD into your terminal or notebook desktop.

**SAY:**
"Now the simulation and control side. We'll walk through how to run a scenario, where to change the parameters, and what the results tell us."

---

### 0:10–0:50 — Running a simulation (terminal vs notebook)

**ON SCREEN:** Show terminal window. Type: `python -m deimos.run_scenario --config configs/scenario_a.yaml`

**SAY:**
"You can run a simulation from the terminal with a single line — just specify a config file. The simulator runs the RK4 integrator, outputs time-history plots, and prints the summary stats to stdout."

**ON SCREEN:** Switch to `explore.ipynb` in the notebook. Show the config cell and `AttitudeSimulator` cell.

**SAY:**
"Or load it in a notebook for faster iteration. Same config files, but you get live plots and can inspect state variables on the fly. Pick whichever fits your workflow."

---

### 0:50–1:20 — Where to change parameters

**ON SCREEN:** Open `src/deimos/constants.py`. Point at key lines.

**SAY:**
"Hardware constants live here in `constants.py` — inertia tensor, wheel limits, max speed, motor specs. These are the plant properties that don't change per scenario."

**ON SCREEN:** Navigate to a config file (e.g., `configs/scenario_a.yaml`).

**SAY:**
"For a specific run, you edit the scenario file: initial attitude, initial rate, disturbance torque. The controller file holds the gains — K, D, and the settling-time target that the gain-sizing math is derived from."

**ON SCREEN:** Show which line to edit (e.g., disturbance torque or initial attitude).

**SAY:**
"Change the initial attitude here, and the scenario runs the same controller against a different starting point. No code touch — pure YAML."

---

### 1:20–2:00 — Simulation results and interpretation

**ON SCREEN:** Scenario A plots — attitude error, angular rate, wheel speed over time.

**SAY:**
"Scenario A is a large slew: 160, 40, 30 degrees initial error, that's 151.66 degrees total. The controller settles under 1 degree in 28.9 seconds. The wheels briefly saturate at t equals zero because that's the largest torque demand — expected behavior. The response stays stable and converges cleanly."

**ON SCREEN:** Scenario B plots — attitude error and control torque under disturbance.

**SAY:**
"Scenario B is disturbance rejection near equilibrium: 8.5 degrees of error while holding against a steady 5-5-5 newton-meter disturbance. Settles in 13.7 seconds, ends within 0.03 degrees, and peak wheel torque never exceeds 5.4 percent of the limit. Momentum drift stays under 2 times 10 to the minus 5 — that's our conservation-law check on the model."

**ON SCREEN:** Show the summary table from the printed output or a bar chart of key metrics.

**SAY:**
"The summary stats tell the story: settling time, control effort, actuator saturation fraction, and momentum drift. These numbers validate that the controller works and that the model is consistent."

---

### 2:00–2:30 — Closing

**ON SCREEN:** End card with repo link or slide showing the five-file workflow.

**SAY:**
"That's the workflow: constants.py for hardware, YAML configs for scenarios, the RK4 simulator for propagation, and the Wie regulator for control. Run it terminal or notebook, change parameters via YAML, read the plots and stats. Full code is in the repo linked below."

---

## Production notes

- ~320 spoken words at natural pace ≈ 2:15–2:30.
- Cold open (no team/project re-intro) — cuts directly from Tulika's CAD segment.
- Show *both* terminal and notebook so viewers see they have options; spend 40 seconds on running, 30 on where params live, 40 on results.
- For the plots, have them pre-rendered so you're not waiting for cells to execute — just pull up the .png files.
- Read the summary stats aloud as you point at them so the numbers stick.
