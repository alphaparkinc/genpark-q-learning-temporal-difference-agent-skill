import sys
import json
from client import TabularQLearningAgent

agent = TabularQLearningAgent(actions=["action_0", "action_1", "action_2"])

def handle_rpc(line):
    global agent
    try:
        req = json.loads(line)
    except Exception:
        return
    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params", {})

    if method == "initialize":
        res = {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "genpark-q-learning-temporal-difference-agent-skill", "version": "1.0.0"},
            "capabilities": {"tools": {}}
        }
    elif method == "tools/list":
        res = {
            "tools": [
                {
                    "name": "select_action",
                    "description": "Select action for current state using epsilon-greedy Q-policy",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"state": {"type": "string"}},
                        "required": ["state"]
                    }
                },
                {
                    "name": "learn_transition",
                    "description": "Perform Bellman temporal difference update Q(s, a) from environment step",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "state": {"type": "string"},
                            "action": {"type": "string"},
                            "reward": {"type": "number"},
                            "next_state": {"type": "string"}
                        },
                        "required": ["state", "action", "reward", "next_state"]
                    }
                }
            ]
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})
        if tool_name == "select_action":
            act = agent.select_action(args.get("state", "default"))
            res = {"content": [{"type": "text", "text": json.dumps({"action": act})}]}
        elif tool_name == "learn_transition":
            agent.learn(
                state=args.get("state"),
                action=args.get("action"),
                reward=args.get("reward"),
                next_state=args.get("next_state")
            )
            res = {"content": [{"type": "text", "text": json.dumps({"status": "success", "q_table": agent.export_q_table()})}]}
        else:
            res = {"isError": True, "content": [{"type": "text", "text": f"Unknown tool {tool_name}"}]}
    else:
        res = {"error": {"code": -32601, "message": "Method not found"}}

    resp = {"jsonrpc": "2.0", "id": req_id, "result": res.get("result", res)}
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()

def main():
    for line in sys.stdin:
        if line.strip():
            handle_rpc(line.strip())

if __name__ == "__main__":
    main()
