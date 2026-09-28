from client import TabularQLearningAgent

def main():
    agent = TabularQLearningAgent(actions=["left", "right"], alpha=0.2)
    for _ in range(50):
        agent.learn("state_A", "right", 5.0, "state_B")
        agent.learn("state_A", "left", -1.0, "state_A")
    table = agent.export_q_table()
    print("Tabular Q-Learning Verification:")
    print(f"Learned Q-Values: {table}")
    print(f"Optimal Action for state_A: {agent.select_action('state_A')}")

if __name__ == "__main__":
    main()
