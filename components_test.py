import numpy as np
import time
import random
import queue
import threading


MODEL_SIZES = {"model_a": 50, "model_b": 100, "model_c": 300, "model_d": 500}
PROMPT_SIZES = {"text": 1, "small_image": 5, "large_image":10}

CVM_CAPACITY = 1000

class EndUser:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def send_request(self):
        interval = np.random.uniform(0, 0.5)  # Random time between 0 and 0.5 seconds
        time.sleep(interval)  # Time to wait between requests sent by user

        # Call to random model with random prompt type
        model = random.choice(list(MODEL_SIZES.keys()))
        prompt_type = random.choice(list(PROMPT_SIZES.keys()))
        prompt_size = PROMPT_SIZES[prompt_type]

        print(f"EndUser: sends request for {model} with {prompt_type} prompt")
        self.scheduler.add_request((model, prompt_type, prompt_size))


class Scheduler:
    def __init__(self, cvms):
        self.cvms = cvms
        self.request_queue = queue.Queue()  # Queue that holds user requests

    def add_request(self, request):
        self.request_queue.put(request)  # Adds request to the end of the queue
        self.process_requests()

    def process_requests(self):
        while not self.request_queue.empty():
            request = self.request_queue.get()  # Request retrieved from the queue
            model_name, prompt_type, prompt_size = request
            dispatched = False

            for cvm in self.cvms:
                if cvm.can_load_request(model_name, prompt_size):
                    print(f"Scheduler: Request {request} sent to {cvm.name}")
                    threading.Thread(target=cvm.process_request, args=(model_name, prompt_size, self), daemon=True).start()
                    dispatched = True
                    break  # Request dispatched, moving on to the next user call

            if not dispatched:
                print(f"Scheduler: not able to dispatch request {request}, requeuing")
                self.request_queue.put(request)
                break

    def decide_unload(self, model_name, cvm):
        # Logic to decide whether to unload the model
        decision = random.choice([True, False])
        print(f"Scheduler: decision to unload model {model_name} from {cvm.name}: {decision}")
        return decision


class CVM:
    def __init__(self, name, allowed_models):
        self.name = name
        self.allowed_models = allowed_models
        self.loaded_models = []
        self.current_load = 0
        self.lock = threading.Lock()

    def can_load_request(self, model_name, prompt_size):
        with self.lock:
            if model_name not in self.allowed_models:  # Check if the model is allowed for the CVM
                return False
            total_size = MODEL_SIZES[model_name] + prompt_size  # Check if it fits
            return self.current_load + total_size <= CVM_CAPACITY

    def process_request(self, model_name, prompt_size, scheduler):
        with self.lock:
            if not self.can_load_request(model_name, prompt_size):
                return

            total_size = MODEL_SIZES[model_name] + prompt_size
            print(f"CVM {self.name}: Loading model {model_name}")
            # Simulated load time - Proportional to the size of the model
            load_time = random.uniform(0.01, 0.03) * MODEL_SIZES[model_name]
            self.loaded_models.append(model_name)
            self.current_load += total_size

        # Perform loading outside the lock
        time.sleep(load_time)
        print(f"CVM {self.name}: Model {model_name} loaded in {load_time:.2f} seconds")

        # Simulated processing time
        processing_time = random.uniform(0.01, 0.03) * prompt_size
        print(f"CVM {self.name}: Processing {model_name} for {processing_time:.2f} seconds")
        time.sleep(processing_time)
        print(f"CVM {self.name}: Completed processing for {model_name}")

        # Perform unloading outside the lock if required
        should_unload = scheduler.decide_unload(model_name, self)
        if should_unload:
            self.unload_model(model_name)

    def unload_model(self, model_name):
        with self.lock:
            if model_name in self.loaded_models:
                print(f"CVM {self.name}: Unloading model {model_name}")
                self.loaded_models.remove(model_name)
                self.current_load -= MODEL_SIZES[model_name]
        # Simulate unloading time outside the lock
        unloading_time = random.uniform(0.01, 0.03) * MODEL_SIZES[model_name] * 0.2
        time.sleep(unloading_time)
        print(f"CVM {self.name}: Model {model_name} unloaded in {unloading_time:.2f} seconds")


# Create CVMs with specific model loading constraints
cvm1 = CVM("CVM1", {"model_a", "model_b"})
cvm2 = CVM("CVM2", {"model_c"})
cvm3 = CVM("CVM3", {"model_d"})

# Initialize Scheduler with CVMs
scheduler = Scheduler([cvm1, cvm2, cvm3])

# Separate threads for multiple users
end_user = EndUser(scheduler)

# Continuously generate requests from the EndUser
def simulate_request():
    while True:
        end_user.send_request()

# Run as a background process
request_thread = threading.Thread(target=simulate_request, daemon=True)
request_thread.start()

# Keep the main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Simulation ended")

