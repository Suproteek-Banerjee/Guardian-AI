class FeedbackHandler:
    def __init__(self):
        self.feedback_log = []

    def collect_feedback(self, observation, action):
        print(f"Observation: {observation}, Action: {action}")
        feedback = input("Enter feedback (-1 for bad, 1 for good, 0 for neutral): ")
        return int(feedback)

    def log_feedback(self, observation, action, feedback):
        self.feedback_log.append({"observation": observation, "action": action, "feedback": feedback})

    def get_feedback_log(self):
        return self.feedback_log
