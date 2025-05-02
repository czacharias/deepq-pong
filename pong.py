import random
import numpy as np
import pygame

from network import predict
pygame.init()

WIDTH, HEIGHT = 700, 500
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PADDLE_WIDTH, PADDLE_HEIGHT = 20, 100
BALL_RADIUS = 7
TRAINING_FPS = 9999999999
GAME_FPS = 60

SCORE_FONT = pygame.font.SysFont("timesnewroman", 50)

class Paddle:
    COLOR = WHITE
    VEL = 5

    def __init__(self, x, y, width, height):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.width = width
        self.height = height
        self.curr_state = 2

    def draw(self, win):
        pygame.draw.rect(win, self.COLOR, (self.x, self.y, self.width, self.height))

    def move(self, dir):
        self.curr_state = dir
        if (dir == 0):
            self.y -= self.VEL
        elif (dir == 1):
            self.y += self.VEL
        else:
            pass

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y

class Ball:
    MAX_VEL = 6
    COLOR = WHITE

    def __init__(self, x, y, radius):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.radius = radius
        self.x_vel = self.MAX_VEL
        self.y_vel = 0

    def draw(self, win):
        pygame.draw.circle(win, self.COLOR, (self.x, self.y), self.radius)

    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
        self.y_vel = 0
        self.x_vel *= -1

class PongGame:
    def __init__(self):
        self.win = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Pong for RL")
        self.clock = pygame.time.Clock()

        self.left_paddle = Paddle(10, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.right_paddle = Paddle(WIDTH - 10 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.ball = Ball(WIDTH // 2, HEIGHT // 2, BALL_RADIUS)

        self.left_score = 0
        self.right_score = 0

        self.memory = []

        self.batch_num = 0
        self.trainer_offset = 0

        self.curr_reward = 0

        self.epsilon = 1
        self.min_epsilon = 0.01  
        self.decay_rate = 0.999
        self.previous_distance = None  


    def draw(self, training_done):
        self.win.fill(BLACK)
        left_score_text = SCORE_FONT.render(f"{self.left_score}", 1, WHITE)
        right_score_text = SCORE_FONT.render(f"{self.right_score}", 1, WHITE)
        self.win.blit(left_score_text, (WIDTH // 4 - left_score_text.get_width() // 2  - 50, 20))
        self.win.blit(right_score_text, (WIDTH * (3 / 4) - right_score_text.get_width() // 2 + 50, 20))
        
        if(not training_done):
            batch_text = SCORE_FONT.render(f"{self.batch_num}", 1, WHITE)
            self.win.blit(batch_text, (WIDTH // 2 - batch_text.get_width() // 2, 20))
        
        for paddle in [self.left_paddle, self.right_paddle]:
            paddle.draw(self.win)

        # for i in range(10, HEIGHT, HEIGHT // 20):
        #     if i % 2 == 1:
        #         continue
        #     pygame.draw.rect(self.win, WHITE, (WIDTH // 2 - 5, i, 10, HEIGHT // 20))

        self.ball.draw(self.win)
        pygame.display.update()

    def handle_collision(self):
        ball = self.ball
        if ball.y + ball.radius >= HEIGHT or ball.y - ball.radius <= 0:
            ball.y_vel *= -1

        if ball.x_vel < 0:
            if self.left_paddle.y <= ball.y <= self.left_paddle.y + self.left_paddle.height:
                if ball.x - ball.radius <= self.left_paddle.x + self.left_paddle.width:
                    ball.x_vel *= -1
                    self._adjust_ball_vel(self.left_paddle)
                    self.trainer_offset = (random.random() * 110 - 55)
                    self.curr_reward += 1
        else:
            if self.right_paddle.y <= ball.y <= self.right_paddle.y + self.right_paddle.height:
                if ball.x + ball.radius >= self.right_paddle.x:
                    ball.x_vel *= -1
                    self.trainer_offset = (random.random() * 110 - 55)
                    self._adjust_ball_vel(self.right_paddle)

    def _adjust_ball_vel(self, paddle):
        middle_y = paddle.y + paddle.height / 2
        difference_in_y = middle_y - self.ball.y
        reduction_factor = (paddle.height / 2) / self.ball.MAX_VEL
        y_vel = difference_in_y / reduction_factor
        self.ball.y_vel = -1 * y_vel

    def handle_paddle_movement(self, keys, model):
        
        action = self.select_action(model)

        still=True
        if action==0 and self.left_paddle.y - self.left_paddle.VEL >= 0:
            self.left_paddle.move(0)
            still = False
        if action==1 and self.left_paddle.y + self.left_paddle.VEL + self.left_paddle.height <= HEIGHT:
            self.left_paddle.move(1)
            still = False
        if still:
            self.left_paddle.move(2)
            
        distance_to_ball = abs((self.left_paddle.y + self.left_paddle.height / 2) - self.ball.y)
        # print(distance_to_ball)
        # print(distance_to_ball / HEIGHT)
        # print((-2/(1 + 3**(-3 * (distance_to_ball / HEIGHT))) + 1.3) * 0.3)
        # self.curr_reward += (-2/(1 + 3**(-3 * (distance_to_ball / HEIGHT))) + 1.3) * 0.3

        # if self.previous_distance is not None:
        #     distance_improvement = self.previous_distance - distance_to_ball
        #     self.curr_reward += distance_improvement * 0.02  

        if self.ball.x_vel < 0: 
            positioning_importance = max(0, 1.0 - (self.ball.x / WIDTH))  

            if distance_to_ball < self.left_paddle.height / 3:  
                self.curr_reward += 0.1 * positioning_importance
            elif distance_to_ball < self.left_paddle.height / 2: 
                self.curr_reward += 0.05 * positioning_importance

            if self.previous_distance is not None:
                distance_improvement = self.previous_distance - distance_to_ball
                if distance_improvement > 0:
                    self.curr_reward += distance_improvement * 0.05 * positioning_importance
        
        if still and self.ball.x_vel < 0 and self.ball.x < WIDTH * 0.7:
            self.curr_reward -= 0.05

        self.previous_distance = distance_to_ball
        

        # if keys[pygame.K_UP] and self.right_paddle.y - self.right_paddle.VEL >= 0:
        #     self.right_paddle.move(0)
        # if keys[pygame.K_DOWN] and self.right_paddle.y + self.right_paddle.VEL + self.right_paddle.height <= HEIGHT:
        #     self.right_paddle.move(1)
        self.right_paddle.y = self.ball.y + self.trainer_offset - self.left_paddle.height //2

    def select_action(self, model):
        if random.random() < self.epsilon: 
            return random.randint(0, 2)
        else:
            ai_actions = predict(model, [self.ball.x / WIDTH, self.ball.y / HEIGHT, self.ball.x_vel / Ball.MAX_VEL, self.ball.y_vel / Ball.MAX_VEL, self.left_paddle.y / HEIGHT, self.right_paddle.y / HEIGHT])
            return np.argmax(ai_actions)

    def remember(self, last_state, reward, done):
        self.memory.append([last_state, self.left_paddle.curr_state, reward, [self.ball.x / WIDTH, self.ball.y / HEIGHT, self.ball.x_vel / Ball.MAX_VEL, self.ball.y_vel / Ball.MAX_VEL, self.left_paddle.y / HEIGHT, self.right_paddle.y / HEIGHT], done])
        if len(self.memory) > 10000:
            self.memory.pop(0)
            
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)


    def play_round(self, model):
        self.ball.reset()
        self.left_paddle.reset()
        self.right_paddle.reset()
        self.previous_distance = None
        self.batch_num += 1
        self.trainer_offset = (random.random() * 80 - 40)

        running = True
        win = 0
        while running:
            self.decay_epsilon()
            self.curr_reward = 0

            self.clock.tick(TRAINING_FPS)
            self.draw(False)

            last_state = [self.ball.x / WIDTH, self.ball.y / HEIGHT, self.ball.x_vel / Ball.MAX_VEL, self.ball.y_vel / Ball.MAX_VEL, self.left_paddle.y / HEIGHT, self.right_paddle.y / HEIGHT]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            keys = pygame.key.get_pressed()
            self.handle_paddle_movement(keys, model)

            self.ball.move()
            self.handle_collision()

            if self.ball.x < 0:
                self.right_score += 1
                running = False
                self.curr_reward -= 2
                win = 0
            elif self.ball.x > WIDTH:
                self.left_score += 1
                running = False
                self.curr_reward += 2
                win = 1
            # else:
            #     self.curr_reward -= 0.001
            self.remember(last_state, self.curr_reward, not running)
        return (self.memory, win)
    
    def main_loop(self, model):
        self.ball.reset()
        self.left_paddle.reset()
        self.right_paddle.reset()
        self.left_score = 0
        self.right_score = 0

        self.epsilon = self.min_epsilon

        running = True
        while running:
            self.clock.tick(GAME_FPS)
            self.draw(True)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False

            keys = pygame.key.get_pressed()
            self.handle_paddle_movement(keys, model)

            self.ball.move()
            self.handle_collision()

            if self.ball.x < 0:
                self.right_score += 1
                self.ball.reset()
                self.left_paddle.reset()
                self.right_paddle.reset()
            elif self.ball.x > WIDTH:
                self.left_score += 1
                self.ball.reset()
                self.left_paddle.reset()
                self.right_paddle.reset()

        pygame.quit()

if __name__ == "__main__":
    game = PongGame()
    #game.play_round(model=None)

    paused = True
    while paused:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                paused = False
    pygame.quit()
