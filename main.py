from flask import Flask, request, redirect, url_for, render_template_string, jsonify
from datetime import datetime
import random

app = Flask(__name__)


# ============================================================
# OOP CLASSES
# ============================================================

class Vehicle:
    total_vehicles = 0

    def __init__(self, vehicle_id, owner, battery, waiting, required_energy, vehicle_type):
        self.vehicle_id = vehicle_id
        self.owner = owner
        self.battery = float(battery)
        self.waiting = int(waiting)
        self.required_energy = float(required_energy)
        self.vehicle_type = vehicle_type
        self.status = "Waiting"
        Vehicle.total_vehicles += 1

    def get_priority(self):
        type_bonus = {
            "Student": 5,
            "Faculty": 10,
            "Emergency": 50,
            "General": 0
        }.get(self.vehicle_type, 0)

        battery_urgency = 100 - self.battery

        return battery_urgency + self.waiting + type_bonus

    def __str__(self):
        return f"{self.vehicle_id} - {self.owner}"


class StudentEV(Vehicle):
    def __init__(self, vehicle_id, owner, battery, waiting, required_energy):
        super().__init__(
            vehicle_id, owner, battery,
            waiting, required_energy, "Student"
        )


class FacultyEV(Vehicle):
    def __init__(self, vehicle_id, owner, battery, waiting, required_energy):
        super().__init__(
            vehicle_id, owner, battery,
            waiting, required_energy, "Faculty"
        )


class EmergencyEV(Vehicle):
    def __init__(self, vehicle_id, owner, battery, waiting, required_energy):
        super().__init__(
            vehicle_id, owner, battery,
            waiting, required_energy, "Emergency"
        )


class ChargingPoint:
    total_points = 0

    def __init__(self, point_id, power):
        self.point_id = point_id
        self.power = float(power)
        self.available = True
        self.current_vehicle = None
        ChargingPoint.total_points += 1

    def allocate(self, vehicle_id):
        if self.available:
            self.available = False
            self.current_vehicle = vehicle_id
            return True
        return False

    def release(self):
        self.available = True
        self.current_vehicle = None

    def __del__(self):
        pass


class ChargingRobot:
    total_robots = 0

    def __init__(self, robot_id, robot_type="Standard", rate=10):
        self.robot_id = robot_id
        self.robot_type = robot_type
        self.rate = float(rate)
        self.available = True
        self.current_vehicle = None
        self.current_point = None
        self.energy_delivered = 0

        ChargingRobot.total_robots += 1

    def assign(self, vehicle, point):
        if self.available and point.available:
            self.available = False
            self.current_vehicle = vehicle.vehicle_id
            self.current_point = point.point_id
            point.allocate(vehicle.vehicle_id)
            vehicle.status = "Charging"
            return True

        return False

    def release(self):
        self.available = True
        self.current_vehicle = None
        self.current_point = None
        self.energy_delivered = 0

    def __del__(self):
        pass


class ChargingSession:
    session_count = 0

    def __init__(self, vehicle, robot, point, energy, duration, tariff):
        ChargingSession.session_count += 1

        self.session_id = f"S{ChargingSession.session_count:03d}"
        self.vehicle = vehicle
        self.robot = robot
        self.point = point
        self.energy = energy
        self.duration = duration
        self.tariff = tariff
        self.cost = energy * tariff
        self.time = datetime.now().strftime("%d-%m-%Y %H:%M")

    # Operator overloading
    def __gt__(self, other):
        return self.energy > other.energy

    def __str__(self):
        return f"{self.session_id} - {self.vehicle}"


class PriorityManager:

    @staticmethod
    def calculate(vehicle):
        return vehicle.get_priority()

    @staticmethod
    def sort(vehicles):
        return sorted(
            vehicles,
            key=lambda x: x.get_priority(),
            reverse=True
        )


# ============================================================
# RL CLASSES
# ============================================================

class PolicyBasedRL:

    def select_action(self, vehicles, robots, points):
        available_robots = [
            r for r in robots if r.available
        ]

        available_points = [
            p for p in points if p.available
        ]

        waiting_vehicles = [
            v for v in vehicles
            if v.status == "Waiting"
        ]

        if not available_robots:
            return None

        if not available_points:
            return None

        if not waiting_vehicles:
            return None

        vehicle = max(
            waiting_vehicles,
            key=lambda v: v.get_priority()
        )

        robot = max(
            available_robots,
            key=lambda r: r.rate
        )

        point = min(
            available_points,
            key=lambda p: p.power
        )

        return robot, vehicle, point

    def calculate_reward(self, vehicle, energy):
        reward = 0

        if vehicle.battery < 30:
            reward += 20

        reward += energy

        reward -= vehicle.waiting * 0.5

        return round(reward, 2)


class ModelBasedRL:

    def predict(self, vehicle, robot, point):
        predicted_energy = min(
            robot.rate,
            vehicle.required_energy
        )

        predicted_waiting = max(
            0,
            vehicle.waiting - 5
        )

        score = (
            vehicle.get_priority()
            + predicted_energy * 2
            - predicted_waiting
            + point.power
        )

        return round(score, 2)

    def best_plan(self, vehicles, robots, points):
        waiting = [
            v for v in vehicles
            if v.status == "Waiting"
        ]

        available_robots = [
            r for r in robots
            if r.available
        ]

        available_points = [
            p for p in points
            if p.available
        ]

        if not waiting or not available_robots or not available_points:
            return None

        best = None
        best_score = -999999

        for vehicle in waiting:
            for robot in available_robots:
                for point in available_points:

                    score = self.predict(
                        vehicle,
                        robot,
                        point
                    )

                    if score > best_score:
                        best_score = score
                        best = (
                            robot,
                            vehicle,
                            point,
                            score
                        )

        return best


class MultiAgentRL:

    def coordinate(self, vehicles, robots, points):
        allocations = []

        waiting = [
            v for v in vehicles
            if v.status == "Waiting"
        ]

        available_robots = [
            r for r in robots
            if r.available
        ]

        available_points = [
            p for p in points
            if p.available
        ]

        waiting = PriorityManager.sort(waiting)

        count = min(
            len(waiting),
            len(available_robots),
            len(available_points)
        )

        for i in range(count):
            allocations.append(
                (
                    available_robots[i],
                    waiting[i],
                    available_points[i]
                )
            )

        return allocations


class HierarchicalLearning:

    def execute(self, vehicles, robots, points):

        # Robot level
        robot_level = len(robots)

        # Coordination level
        coordination_level = min(
            len(vehicles),
            len(robots)
        )

        # Station level
        station_power = sum(
            p.power for p in points
        )

        return {
            "robot_level": robot_level,
            "coordination_level": coordination_level,
            "station_power": station_power
        }


class MetaLearning:

    def adapt(self, condition):

        if condition == "Sudden EV Arrival":
            return "Priority policy updated for sudden EV demand."

        if condition == "Robot Failure":
            return "Remaining robots redistributed charging tasks."

        if condition == "Charger Failure":
            return "EV requests redirected to available chargers."

        if condition == "Power Limitation":
            return "Charging allocation adjusted to available power."

        return "System adapted successfully."


# ============================================================
# STATION
# ============================================================

class ChargingStation:

    def __init__(self):
        self.vehicles = []
        self.robots = []
        self.points = []
        self.sessions = []

        self.total_energy = 0
        self.total_revenue = 0

        self.policy = PolicyBasedRL()
        self.model = ModelBasedRL()
        self.multi_agent = MultiAgentRL()
        self.hierarchical = HierarchicalLearning()
        self.meta = MetaLearning()

        self.total_power = 100

    @property
    def available_power(self):
        used = sum(
            p.power for p in self.points
            if not p.available
        )

        return self.total_power - used


station = ChargingStation()


# ============================================================
# HTML TEMPLATE
# ============================================================

HTML = """

<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>Robot EV Charging</title>

<style>

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    font-family: Arial, sans-serif;
}

body {
    background: #f4f7fb;
    color: #1e293b;
}

/* SIDEBAR */

.sidebar {
    position: fixed;
    left: 0;
    top: 0;
    width: 250px;
    height: 100vh;
    background: #111827;
    color: white;
    padding: 25px 15px;
}

.logo {
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 30px;
    text-align: center;
}

.logo span {
    color: #22c55e;
}

.sidebar a {
    display: block;
    color: #cbd5e1;
    text-decoration: none;
    padding: 13px;
    margin: 5px 0;
    border-radius: 8px;
}

.sidebar a:hover {
    background: #1f2937;
    color: white;
}

/* MAIN */

.main {
    margin-left: 250px;
    padding: 30px;
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 25px;
}

.header h1 {
    font-size: 28px;
}

.badge {
    background: #dcfce7;
    color: #166534;
    padding: 8px 15px;
    border-radius: 20px;
    font-size: 13px;
}

/* CARDS */

.cards {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(190px, 1fr));
    gap: 20px;
    margin-bottom: 25px;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    box-shadow:
        0 3px 12px rgba(0,0,0,0.07);
}

.card h3 {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 10px;
}

.card .number {
    font-size: 30px;
    font-weight: bold;
}

/* SECTIONS */

.section {
    background: white;
    padding: 25px;
    border-radius: 14px;
    margin-bottom: 25px;
    box-shadow:
        0 3px 12px rgba(0,0,0,0.07);
}

.section h2 {
    margin-bottom: 18px;
}

/* FORMS */

.form-grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
}

input, select {
    width: 100%;
    padding: 11px;
    border: 1px solid #cbd5e1;
    border-radius: 7px;
    margin-top: 5px;
}

button {
    border: none;
    padding: 11px 18px;
    border-radius: 7px;
    cursor: pointer;
    background: #2563eb;
    color: white;
    font-weight: bold;
}

button:hover {
    background: #1d4ed8;
}

.green {
    background: #16a34a;
}

.orange {
    background: #ea580c;
}

.red {
    background: #dc2626;
}

.purple {
    background: #7c3aed;
}

/* TABLE */

table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    padding: 12px;
    border-bottom: 1px solid #e2e8f0;
    text-align: left;
}

th {
    background: #f8fafc;
}

/* STATUS */

.status {
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 12px;
}

.available {
    background: #dcfce7;
    color: #166534;
}

.busy {
    background: #fee2e2;
    color: #991b1b;
}

.waiting {
    background: #fef3c7;
    color: #92400e;
}

.charging {
    background: #dbeafe;
    color: #1e40af;
}

/* RL */

.rl-grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(220px, 1fr));
    gap: 15px;
}

.rl-card {
    padding: 20px;
    border-radius: 12px;
    background: #f8fafc;
    border-left: 5px solid #2563eb;
}

.rl-card h3 {
    margin-bottom: 10px;
}

/* MOBILE */

@media(max-width: 800px) {

    .sidebar {
        position: relative;
        width: 100%;
        height: auto;
    }

    .main {
        margin-left: 0;
    }

    .sidebar a {
        display: inline-block;
    }

}

</style>

</head>


<body>

<div class="sidebar">

<div class="logo">
⚡ <span>Robot EV</span>
</div>

<a href="/">🏠 Dashboard</a>

<a href="#vehicles">🚗 EV Management</a>

<a href="#robots">🤖 Robot Management</a>

<a href="#chargers">⚡ Charging Points</a>

<a href="#rl">🧠 Reinforcement Learning</a>

<a href="#sessions">🔋 Charging Sessions</a>

<a href="#reports">📊 Reports</a>

</div>


<div class="main">

<div class="header">

<div>
<h1>Robot-Assisted EV Charging</h1>
<p>Intelligent University Charging Management</p>
</div>

<div class="badge">
● System Online
</div>

</div>


<!-- DASHBOARD CARDS -->

<div class="cards">

<div class="card">
<h3>Total EVs</h3>
<div class="number">
{{ vehicles|length }}
</div>
</div>

<div class="card">
<h3>Charging Robots</h3>
<div class="number">
{{ robots|length }}
</div>
</div>

<div class="card">
<h3>Charging Points</h3>
<div class="number">
{{ points|length }}
</div>
</div>

<div class="card">
<h3>Completed Sessions</h3>
<div class="number">
{{ sessions|length }}
</div>
</div>

<div class="card">
<h3>Energy Used</h3>
<div class="number">
{{ "%.1f"|format(total_energy) }} kWh
</div>
</div>

<div class="card">
<h3>Revenue</h3>
<div class="number">
₹{{ "%.2f"|format(revenue) }}
</div>
</div>

</div>


<!-- EV MANAGEMENT -->

<div class="section" id="vehicles">

<h2>🚗 EV Registration</h2>

<form action="/add_vehicle" method="POST">

<div class="form-grid">

<div>
<label>EV ID</label>
<input name="vehicle_id"
placeholder="EV001" required>
</div>

<div>
<label>Owner Name</label>
<input name="owner"
placeholder="Student Name" required>
</div>

<div>
<label>Battery Level (%)</label>
<input type="number"
name="battery"
min="0"
max="100"
placeholder="30" required>
</div>

<div>
<label>Waiting Time (min)</label>
<input type="number"
name="waiting"
min="0"
placeholder="10" required>
</div>

<div>
<label>Required Energy (kWh)</label>
<input type="number"
step="0.1"
name="energy"
placeholder="20" required>
</div>

<div>
<label>Vehicle Type</label>

<select name="vehicle_type">

<option>Student</option>
<option>Faculty</option>
<option>Emergency</option>
<option>General</option>

</select>

</div>

</div>

<br>

<button class="green">
+ Register EV
</button>

</form>

</div>


<!-- EV TABLE -->

<div class="section">

<h2>Registered EVs</h2>

<table>

<tr>
<th>ID</th>
<th>Owner</th>
<th>Type</th>
<th>Battery</th>
<th>Waiting</th>
<th>Priority</th>
<th>Status</th>
</tr>

{% for v in vehicles %}

<tr>

<td>{{ v.vehicle_id }}</td>

<td>{{ v.owner }}</td>

<td>{{ v.vehicle_type }}</td>

<td>{{ v.battery }}%</td>

<td>{{ v.waiting }} min</td>

<td>
<strong>
{{ "%.1f"|format(v.get_priority()) }}
</strong>
</td>

<td>

<span class="status
{% if v.status == 'Waiting' %}
waiting
{% else %}
charging
{% endif %}">
{{ v.status }}
</span>

</td>

</tr>

{% endfor %}

</table>

</div>


<!-- ROBOTS -->

<div class="section" id="robots">

<h2>🤖 Charging Robots</h2>

<form action="/add_robot" method="POST">

<div class="form-grid">

<div>

<label>Robot ID</label>

<input name="robot_id"
placeholder="R01" required>

</div>

<div>

<label>Robot Type</label>

<select name="robot_type">

<option value="Standard">
Standard Robot
</option>

<option value="Fast">
Fast Charging Robot
</option>

<option value="Emergency">
Emergency Robot
</option>

</select>

</div>

</div>

<br>

<button class="green">
+ Add Robot
</button>

</form>

<br>

<table>

<tr>
<th>Robot</th>
<th>Type</th>
<th>Rate</th>
<th>Status</th>
<th>Current EV</th>
</tr>

{% for r in robots %}

<tr>

<td>{{ r.robot_id }}</td>

<td>{{ r.robot_type }}</td>

<td>{{ r.rate }} kW</td>

<td>

<span class="status
{% if r.available %}
available
{% else %}
busy
{% endif %}">

{% if r.available %}
Available
{% else %}
Busy
{% endif %}

</span>

</td>

<td>
{{ r.current_vehicle or "-" }}
</td>

</tr>

{% endfor %}

</table>

</div>


<!-- CHARGING POINTS -->

<div class="section" id="chargers">

<h2>⚡ Charging Points</h2>

<form action="/add_point" method="POST">

<div class="form-grid">

<div>

<label>Point ID</label>

<input name="point_id"
placeholder="CP01" required>

</div>

<div>

<label>Power Capacity (kW)</label>

<input type="number"
step="0.1"
name="power"
placeholder="20" required>

</div>

</div>

<br>

<button class="green">
+ Add Charging Point
</button>

</form>

<br>

<table>

<tr>
<th>Point</th>
<th>Power</th>
<th>Status</th>
<th>Vehicle</th>
</tr>

{% for p in points %}

<tr>

<td>{{ p.point_id }}</td>

<td>{{ p.power }} kW</td>

<td>

<span class="status
{% if p.available %}
available
{% else %}
busy
{% endif %}">

{% if p.available %}
Available
{% else %}
Occupied
{% endif %}

</span>

</td>

<td>
{{ p.current_vehicle or "-" }}
</td>

</tr>

{% endfor %}

</table>

</div>


<!-- RL SECTION -->

<div class="section" id="rl">

<h2>🧠 Reinforcement Learning Control</h2>

<div class="rl-grid">

<div class="rl-card">

<h3>🎯 Policy-Based RL</h3>

<p>
Selects the best immediate EV,
robot and charging point using
current state and priority.
</p>

<br>

<form action="/policy" method="POST">

<button>
Run Policy Allocation
</button>

</form>

</div>


<div class="rl-card">

<h3>🔮 Model-Based RL</h3>

<p>
Predicts future charging conditions
and evaluates possible allocation
plans.
</p>

<br>

<form action="/model" method="POST">

<button class="purple">
Run Future Planning
</button>

</form>

</div>


<div class="rl-card">

<h3>🤖 Multi-Agent RL</h3>

<p>
Multiple charging robots coordinate
their actions to avoid resource
conflicts.
</p>

<br>

<form action="/multi_agent" method="POST">

<button class="orange">
Run Multi-Agent Allocation
</button>

</form>

</div>


<div class="rl-card">

<h3>🏢 Hierarchical Learning</h3>

<p>
Robot level → Coordination level
→ Station level decision making.
</p>

<br>

<form action="/hierarchical" method="POST">

<button>
Run Hierarchical Control
</button>

</form>

</div>

</div>

</div>


<!-- ACTIONS -->

<div class="section">

<h2>⚙️ Charging Operations</h2>

<form action="/start_charging" method="POST"
style="display:inline">

<button class="green">
🔋 Start Charging
</button>

</form>

&nbsp;

<form action="/complete" method="POST"
style="display:inline">

<button class="orange">
✓ Complete Session
</button>

</form>

&nbsp;

<form action="/unexpected" method="POST"
style="display:inline">

<select name="condition"
style="width:auto;display:inline-block">

<option>Sudden EV Arrival</option>
<option>Robot Failure</option>
<option>Charger Failure</option>
<option>Power Limitation</option>

</select>

<button class="red">
⚠ Adapt System
</button>

</form>

</div>


<!-- SESSIONS -->

<div class="section" id="sessions">

<h2>🔋 Charging Sessions</h2>

<table>

<tr>

<th>Session</th>
<th>EV</th>
<th>Robot</th>
<th>Point</th>
<th>Energy</th>
<th>Tariff</th>
<th>Cost</th>
<th>Time</th>

</tr>

{% for s in sessions %}

<tr>

<td>{{ s.session_id }}</td>

<td>{{ s.vehicle }}</td>

<td>{{ s.robot }}</td>

<td>{{ s.point }}</td>

<td>{{ "%.2f"|format(s.energy) }} kWh</td>

<td>₹{{ "%.2f"|format(s.tariff) }}</td>

<td>
<strong>
₹{{ "%.2f"|format(s.cost) }}
</strong>
</td>

<td>{{ s.time }}</td>

</tr>

{% endfor %}

</table>

</div>


<!-- REPORT -->

<div class="section" id="reports">

<h2>📊 Station Report</h2>

<div class="cards">

<div class="card">

<h3>Available Power</h3>

<div class="number">
{{ "%.1f"|format(available_power) }} kW
</div>

</div>

<div class="card">

<h3>Station Capacity</h3>

<div class="number">
{{ total_power }} kW
</div>

</div>

<div class="card">

<h3>Utilization</h3>

<div class="number">
{{ "%.1f"|format(utilization) }}%
</div>

</div>

<div class="card">

<h3>RL Agents</h3>

<div class="number">
{{ robots|length }}
</div>

</div>

</div>

</div>


<!-- FOOTER -->

<div style="
text-align:center;
padding:25px;
color:#64748b;
">

Robot-Assisted Intelligent EV Charging System
<br>
Policy RL • Model RL • MARL • Hierarchical Learning • Meta-Learning

</div>

</div>

</body>

</html>

"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():

    occupied = sum(
        1 for p in station.points
        if not p.available
    )

    if station.points:
        utilization = (
            occupied /
            len(station.points)
        ) * 100
    else:
        utilization = 0

    return render_template_string(
        HTML,
        vehicles=station.vehicles,
        robots=station.robots,
        points=station.points,
        sessions=station.sessions,
        total_energy=station.total_energy,
        revenue=station.total_revenue,
        available_power=station.available_power,
        total_power=station.total_power,
        utilization=utilization
    )


# ============================================================
# ADD VEHICLE
# ============================================================

@app.route("/add_vehicle", methods=["POST"])
def add_vehicle():

    vehicle_id = request.form["vehicle_id"]
    owner = request.form["owner"]
    battery = request.form["battery"]
    waiting = request.form["waiting"]
    energy = request.form["energy"]
    vehicle_type = request.form["vehicle_type"]

    if vehicle_type == "Student":

        vehicle = StudentEV(
            vehicle_id,
            owner,
            battery,
            waiting,
            energy
        )

    elif vehicle_type == "Faculty":

        vehicle = FacultyEV(
            vehicle_id,
            owner,
            battery,
            waiting,
            energy
        )

    elif vehicle_type == "Emergency":

        vehicle = EmergencyEV(
            vehicle_id,
            owner,
            battery,
            waiting,
            energy
        )

    else:

        vehicle = Vehicle(
            vehicle_id,
            owner,
            battery,
            waiting,
            energy,
            "General"
        )

    station.vehicles.append(vehicle)

    return redirect(url_for("index"))


# ============================================================
# ADD ROBOT
# ============================================================

@app.route("/add_robot", methods=["POST"])
def add_robot():

    robot_id = request.form["robot_id"]
    robot_type = request.form["robot_type"]

    if robot_type == "Fast":
        robot = ChargingRobot(
            robot_id,
            "Fast",
            20
        )

    elif robot_type == "Emergency":
        robot = ChargingRobot(
            robot_id,
            "Emergency",
            25
        )

    else:
        robot = ChargingRobot(
            robot_id,
            "Standard",
            10
        )

    station.robots.append(robot)

    return redirect(url_for("index"))


# ============================================================
# ADD CHARGING POINT
# ============================================================

@app.route("/add_point", methods=["POST"])
def add_point():

    point_id = request.form["point_id"]
    power = request.form["power"]

    point = ChargingPoint(
        point_id,
        power
    )

    station.points.append(point)

    return redirect(url_for("index"))


# ============================================================
# POLICY BASED RL
# ============================================================

@app.route("/policy", methods=["POST"])
def policy():

    result = station.policy.select_action(
        station.vehicles,
        station.robots,
        station.points
    )

    if result:

        robot, vehicle, point = result

        robot.assign(
            vehicle,
            point
        )

    return redirect(url_for("index"))


# ============================================================
# MODEL BASED RL
# ============================================================

@app.route("/model", methods=["POST"])
def model():

    result = station.model.best_plan(
        station.vehicles,
        station.robots,
        station.points
    )

    if result:

        robot, vehicle, point, score = result

        print(
            f"MODEL PLAN: {robot.robot_id} "
            f"-> {vehicle.vehicle_id} "
            f"-> {point.point_id} "
            f"Score={score}"
        )

    return redirect(url_for("index"))


# ============================================================
# MULTI AGENT RL
# ============================================================

@app.route("/multi_agent", methods=["POST"])
def multi_agent():

    allocations = station.multi_agent.coordinate(
        station.vehicles,
        station.robots,
        station.points
    )

    for robot, vehicle, point in allocations:

        robot.assign(
            vehicle,
            point
        )

    return redirect(url_for("index"))


# ============================================================
# HIERARCHICAL
# ============================================================

@app.route("/hierarchical", methods=["POST"])
def hierarchical():

    result = station.hierarchical.execute(
        station.vehicles,
        station.robots,
        station.points
    )

    print("Hierarchical Result:", result)

    return redirect(url_for("index"))


# ============================================================
# START CHARGING
# ============================================================

@app.route("/start_charging", methods=["POST"])
def start_charging():

    for robot in station.robots:

        if not robot.available:

            vehicle = next(
                (
                    v for v in station.vehicles
                    if v.vehicle_id ==
                    robot.current_vehicle
                ),
                None
            )

            if vehicle:

                energy = min(
                    robot.rate,
                    vehicle.required_energy
                )

                vehicle.required_energy -= energy

                vehicle.battery = min(
                    100,
                    vehicle.battery +
                    energy * 2
                )

                robot.energy_delivered += energy

                reward = station.policy.calculate_reward(
                    vehicle,
                    energy
                )

                print(
                    f"Robot {robot.robot_id} "
                    f"Reward = {reward}"
                )

    return redirect(url_for("index"))


# ============================================================
# COMPLETE SESSION
# ============================================================

@app.route("/complete", methods=["POST"])
def complete():

    active_robots = [
        r for r in station.robots
        if not r.available
    ]

    for robot in active_robots:

        vehicle = next(
            (
                v for v in station.vehicles
                if v.vehicle_id ==
                robot.current_vehicle
            ),
            None
        )

        point = next(
            (
                p for p in station.points
                if p.point_id ==
                robot.current_point
            ),
            None
        )

        if not vehicle or not point:
            continue

        energy = robot.energy_delivered

        if energy <= 0:
            energy = robot.rate

        hour = datetime.now().hour

        if 18 <= hour < 22:

            tariff = 10.0

        else:

            tariff = 6.0

        session = ChargingSession(
            vehicle.vehicle_id,
            robot.robot_id,
            point.point_id,
            energy,
            30,
            tariff
        )

        station.sessions.append(session)

        station.total_energy += energy

        station.total_revenue += session.cost

        vehicle.status = "Completed"

        point.release()

        robot.release()

    return redirect(url_for("index"))


# ============================================================
# UNEXPECTED CONDITIONS / META LEARNING
# ============================================================

@app.route("/unexpected", methods=["POST"])
def unexpected():

    condition = request.form["condition"]

    message = station.meta.adapt(
        condition
    )

    print(
        f"META LEARNING: {message}"
    )

    if condition == "Robot Failure":

        busy = [
            r for r in station.robots
            if not r.available
        ]

        if busy:

            failed_robot = random.choice(
                busy
            )

            print(
                f"Robot failure simulated: "
                f"{failed_robot.robot_id}"
            )

    elif condition == "Charger Failure":

        available = [
            p for p in station.points
            if p.available
        ]

        if available:

            failed_point = random.choice(
                available
            )

            failed_point.available = False

            print(
                f"Charger failure simulated: "
                f"{failed_point.point_id}"
            )

    elif condition == "Power Limitation":

        station.total_power = max(
            20,
            station.total_power - 20
        )

        print(
            f"Power reduced to "
            f"{station.total_power} kW"
        )

    return redirect(url_for("index"))


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print(" ROBOT-ASSISTED INTELLIGENT EV CHARGING SYSTEM")
    print("=" * 60)

    print("\nStarting Flask Web Server...")

    print(
        "Open: http://127.0.0.1:5000"
    )

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )