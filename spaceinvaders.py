#!/usr/bin/env python

# Space Invaders Environment for AI
# Base originale di Lee Robinson - Adattato per Data Science Project

import pygame
import sys
import os
import math
import csv
import numpy as np
from pygame.locals import *
from os.path import abspath, dirname
from random import choice, randint

# --- CONFIGURAZIONE ---
BASE_PATH = abspath(dirname(_file_))
FONT_PATH = BASE_PATH + '/fonts/'
IMAGE_PATH = BASE_PATH + '/images/'
SOUND_PATH = BASE_PATH + '/sounds/'

# Configurazione Visualizzazione
WATCH_MODE = False      # True: Vedi il gioco. False: Training veloce (schermo nero).
COLLECT_DATA = False   # Impostalo a True nel tuo script di training, non qui.

# Costanti Colori
WHITE = (255, 255, 255)
GREEN = (78, 255, 87)
YELLOW = (241, 255, 0)
BLUE = (80, 255, 239)
PURPLE = (203, 0, 255)
RED = (237, 28, 36)

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
FONT = FONT_PATH + 'space_invaders.ttf'
IMG_NAMES = ['ship', 'mystery',
             'enemy1_1', 'enemy1_2',
             'enemy2_1', 'enemy2_2',
             'enemy3_1', 'enemy3_2',
             'explosionblue', 'explosiongreen', 'explosionpurple',
             'laser', 'enemylaser']

# Caricamento Immagini Sicuro
try:
    IMAGES = {name: pygame.image.load(IMAGE_PATH + '{}.png'.format(name)).convert_alpha()
              for name in IMG_NAMES}
except Exception as e:
    print(f"Errore caricamento immagini: {e}. Assicurati che la cartella 'images' sia presente.")
    sys.exit()

BLOCKERS_POSITION = 450
ENEMY_DEFAULT_POSITION = 65
ENEMY_MOVE_DOWN = 35


# --- CLASSI SPRITE (Entità del gioco) ---

class Ship(pygame.sprite.Sprite):
    def _init_(self):
        pygame.sprite.Sprite._init_(self)
        self.image = IMAGES['ship']
        self.rect = self.image.get_rect(topleft=(375, 540))
        self.speed = 5

    def update(self, *args):
        SCREEN.blit(self.image, self.rect)

class Life(pygame.sprite.Sprite):
    def _init_(self, xpos, ypos):
        pygame.sprite.Sprite._init_(self)
        self.image = IMAGES['ship']
        self.image = pygame.transform.scale(self.image, (23, 23))
        self.rect = self.image.get_rect(topleft=(xpos, ypos))

    def update(self, *args):
        pass

class Bullet(pygame.sprite.Sprite):
    def _init_(self, xpos, ypos, direction, speed, filename, side):
        pygame.sprite.Sprite._init_(self)
        self.image = IMAGES[filename]
        self.rect = self.image.get_rect(topleft=(xpos, ypos))
        self.speed = speed
        self.direction = direction
        self.side = side
        self.filename = filename

    def update(self, *args):
        self.rect.y += self.speed * self.direction


class Enemy(pygame.sprite.Sprite):
    def _init_(self, row, column):
        pygame.sprite.Sprite._init_(self)
        self.row = row
        self.column = column
        self.images = []
        self.load_images()
        self.index = 0
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()

    def toggle_image(self):
        self.index += 1
        if self.index >= len(self.images):
            self.index = 0
        self.image = self.images[self.index]

    def update(self, *args):
        pass

    def load_images(self):
        images = {0: ['1_2', '1_1'], 1: ['2_2', '2_1'], 2: ['2_2', '2_1'],
                  3: ['3_1', '3_2'], 4: ['3_1', '3_2']}
        img1, img2 = (IMAGES['enemy{}'.format(img_num)] for img_num in images[self.row])
        self.images.append(pygame.transform.scale(img1, (40, 35)))
        self.images.append(pygame.transform.scale(img2, (40, 35)))

class EnemiesGroup(pygame.sprite.Group):
    def _init_(self, columns, rows, enemyPosition):
        pygame.sprite.Group._init_(self)
        self.enemies = [[None] * columns for _ in range(rows)]
        self.columns = columns
        self.rows = rows
        self.leftAddMove = 0
        self.rightAddMove = 0
        self.moveTime = 600
        self.direction = 1
        self.rightMoves = 30
        self.leftMoves = 30
        self.moveNumber = 15
        self.timer = pygame.time.get_ticks()
        self.bottom = enemyPosition + ((rows - 1) * 45) + 35
        self._aliveColumns = list(range(columns))
        self._leftAliveColumn = 0
        self._rightAliveColumn = columns - 1

    def update(self, current_time):
        if current_time - self.timer > self.moveTime:
            if self.direction == 1:
                max_move = self.rightMoves + self.rightAddMove
            else:
                max_move = self.leftMoves + self.leftAddMove

            if self.moveNumber >= max_move:
                self.leftMoves = 30 + self.rightAddMove
                self.rightMoves = 30 + self.leftAddMove
                self.direction *= -1
                self.moveNumber = 0
                self.bottom = 0
                for enemy in self:
                    enemy.rect.y += ENEMY_MOVE_DOWN
                    enemy.toggle_image()
                    if self.bottom < enemy.rect.y + 35:
                        self.bottom = enemy.rect.y + 35
            else:
                velocity = 10 if self.direction == 1 else -10
                for enemy in self:
                    enemy.rect.x += velocity
                    enemy.toggle_image()
                self.moveNumber += 1
            self.timer += self.moveTime

    def add_internal(self, *sprites):
        super(EnemiesGroup, self).add_internal(*sprites)
        for s in sprites:
            self.enemies[s.row][s.column] = s

    def remove_internal(self, *sprites):
        super(EnemiesGroup, self).remove_internal(*sprites)
        for s in sprites:
            self.kill(s)
        self.update_speed()

    def is_column_dead(self, column):
        return not any(self.enemies[row][column] for row in range(self.rows))

    def random_bottom(self):
        col = choice(self._aliveColumns)
        col_enemies = (self.enemies[row - 1][col] for row in range(self.rows, 0, -1))
        return next((en for en in col_enemies if en is not None), None)

    def update_speed(self):
       pass

    def kill(self, enemy):
        self.enemies[enemy.row][enemy.column] = None
        is_column_dead = self.is_column_dead(enemy.column)
        if is_column_dead:
            self._aliveColumns.remove(enemy.column)
        if enemy.column == self._rightAliveColumn:
            while self._rightAliveColumn > 0 and is_column_dead:
                self._rightAliveColumn -= 1
                self.rightAddMove += 5
                is_column_dead = self.is_column_dead(self._rightAliveColumn)
        elif enemy.column == self._leftAliveColumn:
            while self._leftAliveColumn < self.columns and is_column_dead:
                self._leftAliveColumn += 1
                self.leftAddMove += 5
                is_column_dead = self.is_column_dead(self._leftAliveColumn)

class Blocker(pygame.sprite.Sprite):
    def _init_(self, size, color, row, column):
        pygame.sprite.Sprite._init_(self)
        self.height = size
        self.width = size
        self.color = color
        self.image = pygame.Surface((self.width, self.height))
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self.row = row
        self.column = column

    def update(self, *args):
        pass

class Mystery(pygame.sprite.Sprite):
    def _init_(self):
        pygame.sprite.Sprite._init_(self)
        self.image = IMAGES['mystery']
        self.image = pygame.transform.scale(self.image, (75, 35))
        self.rect = self.image.get_rect(topleft=(-80, 45))
        self.row = 5
        self.moveTime = 25000
        self.direction = 1
        self.timer = pygame.time.get_ticks()
        self.mysteryEntered = pygame.mixer.Sound(SOUND_PATH + 'mysteryentered.wav')
        self.mysteryEntered.set_volume(0.3)
        self.playSound = True

    def update(self, currentTime, *args):
        resetTimer = False
        passed = currentTime - self.timer
        if passed > self.moveTime:
            if (self.rect.x < 0 or self.rect.x > 800) and self.playSound:
                self.mysteryEntered.play()
                self.playSound = False
            if self.rect.x < 840 and self.direction == 1:
                self.mysteryEntered.fadeout(4000)
                self.rect.x += 2
            if self.rect.x > -100 and self.direction == -1:
                self.mysteryEntered.fadeout(4000)
                self.rect.x -= 2

        if self.rect.x > 830:
            self.playSound = True
            self.direction = -1
            resetTimer = True
        if self.rect.x < -90:
            self.playSound = True
            self.direction = 1
            resetTimer = True
        if passed > self.moveTime and resetTimer:
            self.timer = currentTime

class EnemyExplosion(pygame.sprite.Sprite):
    def _init_(self, enemy, *groups):
        super(EnemyExplosion, self)._init_(*groups)
        self.image = pygame.transform.scale(self.get_image(enemy.row), (40, 35))
        self.image2 = pygame.transform.scale(self.get_image(enemy.row), (50, 45))
        self.rect = self.image.get_rect(topleft=(enemy.rect.x, enemy.rect.y))
        self.timer = pygame.time.get_ticks()

    @staticmethod
    def get_image(row):
        img_colors = ['purple', 'blue', 'blue', 'green', 'green']
        return IMAGES['explosion{}'.format(img_colors[row])]

    def update(self, current_time, *args):
        passed = current_time - self.timer
        if passed <= 100:
            SCREEN.blit(self.image, self.rect)
        elif passed <= 200:
            SCREEN.blit(self.image2, (self.rect.x - 6, self.rect.y - 6))
        elif 400 < passed:
            self.kill()

class Text(object):
    def _init_(self, textFont, size, message, color, xpos, ypos):
        self.font = pygame.font.Font(textFont, size)
        self.surface = self.font.render(message, True, color)
        self.rect = self.surface.get_rect(topleft=(xpos, ypos))
    def draw(self, surface):
        surface.blit(self.surface, self.rect)


# --- AMBIENTE DI GIOCO PER AI ---

class SpaceInvadersEnvironment(object):
    def _init_(self, collect_data=False):
        pygame.mixer.pre_init(44100, -16, 1, 4096)
        pygame.init()
        self.clock = pygame.time.Clock()
        self.caption = pygame.display.set_caption('Space Invaders - AI Environment')
        self.background = pygame.image.load(IMAGE_PATH + 'background.jpg').convert()
        self.enemyPosition = ENEMY_DEFAULT_POSITION
        self.score = 0
        self.collect_data = collect_data
        self.simulated_time = pygame.time.get_ticks() # Sincronizza l'inizio

        self.create_audio()

        # CSV Writer per Data Collection
        if self.collect_data:
            self.log_file = open('dataset_space_invaders.csv', 'w', newline='')
            self.writer = csv.writer(self.log_file)
            # Intestazione: Stato (6 valori) + Azione
            self.writer.writerow(["p_x", "e_x", "e_y", "b_x", "b_y", "dir", "ACTION"])

        self.reset()

    def create_audio(self):
        # 1. Inizializza SEMPRE le variabili di tempo (servono alla logica di gioco)
        self.noteIndex = 0
        self.noteTimer = pygame.time.get_ticks()

        # 2. Se è Training (WATCH_MODE False), usa i suoni finti ed esci
        if not WATCH_MODE:
            class DummySound:
                def play(self): pass
                def set_volume(self, v): pass
                def stop(self): pass
                def fadeout(self, t): pass

            # Crea dizionario con suoni finti
            self.sounds = {name: DummySound() for name in ['shoot', 'shoot2', 'invaderkilled', 'mysterykilled', 'shipexplosion']}
            # Crea lista note finte
            self.musicNotes = [DummySound() for _ in range(4)]
            return # Esce qui, ma ora noteTimer esiste già!

        # 3. Se è Watch Mode (True), carica i suoni veri dal disco
        self.sounds = {}
        for sound_name in ['shoot', 'shoot2', 'invaderkilled', 'mysterykilled', 'shipexplosion']:
            self.sounds[sound_name] = pygame.mixer.Sound(SOUND_PATH + '{}.wav'.format(sound_name))
            self.sounds[sound_name].set_volume(0.2)

        self.musicNotes = [pygame.mixer.Sound(SOUND_PATH + '{}.wav'.format(i)) for i in range(4)]
        for sound in self.musicNotes:
            sound.set_volume(0.5)

    def play_main_music(self, currentTime):
        # Suona le note a tempo con il movimento dei nemici
        if currentTime - self.noteTimer > self.enemies.moveTime:
            self.note = self.musicNotes[self.noteIndex]
            if self.noteIndex < 3:
                self.noteIndex += 1
            else:
                self.noteIndex = 0

            self.note.play()
            self.noteTimer += self.enemies.moveTime

    def reset(self):
        # Reset totale dell'ambiente
        self.current_step = 0

        # Inizia facile
        self.level = 1
        self.base_move_time = 800 # Millisecondi tra un movimento e l'altro (800 = Lento)

        self.simulated_time = pygame.time.get_ticks()
        self.enemyPosition = ENEMY_DEFAULT_POSITION
        self.player = Ship()
        self.playerGroup = pygame.sprite.Group(self.player)
        self.explosionsGroup = pygame.sprite.Group()
        self.bullets = pygame.sprite.Group()
        self.mysteryShip = Mystery()
        self.mysteryGroup = pygame.sprite.Group(self.mysteryShip)
        self.enemyBullets = pygame.sprite.Group()

        # Reset Nemici
        self.make_enemies()
        self.enemies.moveTime = self.base_move_time

        # Reset Blockers
        self.make_blockers()

        # Reset Vite (3 vite: life1, life2, life3)
        self.life1 = Life(715, 3)
        self.life2 = Life(742, 3)
        self.life3 = Life(769, 3)
        self.livesGroup = pygame.sprite.Group(self.life1, self.life2, self.life3)

        self.allSprites = pygame.sprite.Group(self.player, self.enemies, self.livesGroup, self.mysteryGroup)

        self.timer = pygame.time.get_ticks()
        self.shipTimer = pygame.time.get_ticks()
        self.score = 0
        self.gameOver = False
        self.makeNewShip = False
        self.shipAlive = True

        return self.get_state()

    def make_blockers(self):
        self.allBlockers = pygame.sprite.Group()
        # Ciclo per creare 4 strutture separate (0, 1, 2, 3)
        for number in range(4):
            for row in range(4):
                for column in range(9):
                    blocker = Blocker(10, GREEN, row, column)
                    # Posizione X calcolata in base al numero del blocco (0-3)
                    blocker.rect.x = 50 + (200 * number) + (column * blocker.width)
                    blocker.rect.y = BLOCKERS_POSITION + (row * blocker.height)
                    self.allBlockers.add(blocker)

    def make_enemies(self):
        self.enemies = EnemiesGroup(10, 5, self.enemyPosition)
        for row in range(5):
            for column in range(10):
                enemy = Enemy(row, column)
                enemy.rect.x = 157 + (column * 50)
                enemy.rect.y = self.enemyPosition + (row * 45)
                self.enemies.add(enemy)

    def get_state(self):
        """
        OCCHI DELL'IA: Restituisce un vettore normalizzato (0-1)
        """
        # 1. Player X
        p_x = self.player.rect.x / 800.0 if self.shipAlive else 0.5

        # 2. Nemico più vicino
        closest_e_x, closest_e_y = 0.0, 0.0
        min_dist = 9999
        if self.enemies:
            for row in self.enemies.enemies:
                for enemy in row:
                    if enemy:
                        d = math.sqrt((self.player.rect.x - enemy.rect.x)*2 + (self.player.rect.y - enemy.rect.y)*2)
                        if d < min_dist:
                            min_dist = d
                            closest_e_x = enemy.rect.x / 800.0
                            closest_e_y = enemy.rect.y / 600.0

        # Aggiungiamo lo stato del NOSTRO proiettile
        # 1.0 se abbiamo un colpo in volo, 0.0 se siamo liberi di sparare
        my_bullet_active = 1.0 if len(self.bullets) > 0 else 0.0

        # 3. Proiettile nemico più vicino
        closest_b_x, closest_b_y = 0.0, 0.0
        min_b_dist = 9999
        for b in self.enemyBullets:
            d = math.sqrt((self.player.rect.x - b.rect.x)*2 + (self.player.rect.y - b.rect.y)*2)
            if d < min_b_dist:
                min_b_dist = d
                closest_b_x = b.rect.x / 800.0
                closest_b_y = b.rect.y / 600.0

        # 4. Direzione Nemici
        direction = 1.0 if self.enemies.direction == 1 else -1.0

        return np.array([p_x, my_bullet_active, closest_e_x, closest_e_y, closest_b_x, closest_b_y, direction], dtype=np.float32)

    def step(self, action):
        """
        MANI DELL'IA: Esegue azione e restituisce (stato, reward, done)
        Action: 0=Fermo, 1=SX, 2=DX, 3=Spara
        """
        reward = 0
        done = False

        # --- GESTIONE TEMPO SIMULATO ---
        if WATCH_MODE:
            # Se guardiamo la partita, usiamo il tempo reale
            current_time = pygame.time.get_ticks()
        else:
            # Se alleniamo, SIMULIAMO il tempo (16ms = ~60 FPS)
            # Questo assicura che ogni step sia esattamente un frame diverso
            self.simulated_time += 16
            current_time = self.simulated_time

        # --- 0. RESPAWN LOGIC ---
        if self.makeNewShip and (current_time - self.shipTimer > 900):
            self.player = Ship()
            self.allSprites.add(self.player)
            self.playerGroup.add(self.player)
            self.makeNewShip = False
            self.shipAlive = True

        # --- 1. ESECUZIONE AZIONE ---
        if self.shipAlive:
            if action == 1: # Left
                if self.player.rect.x > 10: self.player.rect.x -= self.player.speed
            elif action == 2: # Right
                if self.player.rect.x < 740: self.player.rect.x += self.player.speed
            elif action == 3: # Shoot
                if len(self.bullets) == 0:
                    self.bullets.add(Bullet(self.player.rect.x + 23, self.player.rect.y + 5, -1, 15, 'laser', 'center'))
                    self.sounds['shoot'].play()
                else:
                    reward -= 0.1 # Piccola penalità spam

        # --- 2. FISICA DEL GIOCO ---
        self.play_main_music(current_time)
        self.enemies.update(current_time)
        self.bullets.update()
        self.enemyBullets.update(current_time)
        self.mysteryGroup.update(current_time)
        self.explosionsGroup.update(current_time)
        self.livesGroup.update()

       # --- FUOCO NEMICO "PROGRESSIVE EVIL MODE" ---

        # 1. Calcola quanti proiettili possono esserci a schermo in base al livello
        # Livello 1: 3 proiettili
        # Livello 2: 4 proiettili
        # ...
        # Livello 7+: 10 proiettili (Cap massimo)
        max_allowed_bullets = min(10, 2 + self.level)

        # 2. Probabilità di sparo (parte bassa, cresce col livello)
        # Livello 1: 2%
        # Livello 10: ~20%
        current_shoot_prob = 0.01 + (self.level * 0.02)
        if current_shoot_prob > 0.25: current_shoot_prob = 0.25

        # 3. Logica di sparo
        # Spara SOLO se non abbiamo raggiunto il limite del livello corrente
        if len(self.enemyBullets) < max_allowed_bullets and np.random.rand() < current_shoot_prob:
             if len(self.enemies) > 0:
                shooter = np.random.choice(self.enemies.sprites())
                bullet = Bullet(shooter.rect.x + 14, shooter.rect.y + 20, 1, 5, 'enemylaser', 'center')
                self.enemyBullets.add(bullet)
                self.allSprites.add(bullet)

        # --- 3. GESTIONE COLLISIONI & REWARD ---

        # A. Colpito Nemico (+10)
        hits = pygame.sprite.groupcollide(self.enemies, self.bullets, True, True)
        for enemy in hits:
            self.sounds['invaderkilled'].play()
            reward += 10
            EnemyExplosion(enemy, self.explosionsGroup)

        # B. Colpito Mystery (+50)
        if pygame.sprite.groupcollide(self.mysteryGroup, self.bullets, True, True):
            reward += 50
            self.mysteryShip = Mystery() # Respawn mystery
            self.mysteryGroup.add(self.mysteryShip)

        # C. Proiettile mancato (-1)
        for b in self.bullets:
            if b.rect.y < 0:
                reward -= 1
                b.kill()

        # Pulizia Proiettili Nemici (Usciti dallo schermo in basso)
        for b in self.enemyBullets:
            if b.rect.y > 600:
                b.kill()

        # Gestione Blocker (Muri):
        # I proiettili (sia tuoi che nemici) distruggono i blocker e si distruggono a vicenda
        pygame.sprite.groupcollide(self.bullets, self.allBlockers, True, True)
        pygame.sprite.groupcollide(self.enemyBullets, self.allBlockers, True, True)

        # Se i nemici scendono troppo, mangiano i blocker
        if self.enemies.bottom >= BLOCKERS_POSITION:
            pygame.sprite.groupcollide(self.enemies, self.allBlockers, False, True)

        # D. COLLISIONE PLAYER (Gestione Vite)
        hits_player = pygame.sprite.groupcollide(self.playerGroup, self.enemyBullets, True, True)

        if hits_player or pygame.sprite.groupcollide(self.enemies, self.playerGroup, False, True):
            self.sounds['shipexplosion'].play()

            # Controlla vite rimanenti
            if len(self.livesGroup) > 0:
                # HIT: Perde una vita, ma continua (-50)
                reward -= 50
                life_to_remove = self.livesGroup.sprites()[-1]
                life_to_remove.kill()

                self.makeNewShip = True
                self.shipTimer = current_time
                self.shipAlive = False

                # --- FIX PROIETTILI CONGELATI ---
                # Non usare .empty(), ma .kill() su ogni proiettile!
                for bullet in self.enemyBullets:
                    bullet.kill() # Rimuove sia da enemyBullets che da allSprites

                # Puliamo anche i proiettili nostri per pulizia
                for bullet in self.bullets:
                    bullet.kill()
            else:
                # DIE: Game Over (-100)
                reward -= 100
                done = True

        # E. Nemici toccano il fondo (-100)
        if self.enemies.bottom >= 540:
            reward -= 100
            done = True

        # --- MODIFICA: VITTORIA LIVELLO (Progressione Infinita) ---
        if not self.enemies:
            # 1. PREMIO ENORME
            reward += 50

            # 2. AUMENTA DIFFICOLTÀ
            self.level += 1

            # VELOCITÀ ESPONENZIALE
            # Parte da 800.
            # Lvl 2: 600ms
            # Lvl 3: 400ms
            # Lvl 4: 200ms
            # Lvl 5+: 50ms (Schegge impazzite)
            new_move_time = max(50, 800 - ((self.level -1) * 200))

            # 4. POSIZIONE: Scendono di 35px ogni livello (reset ogni 5 livelli)
            self.enemyPosition = ENEMY_DEFAULT_POSITION + (35 * ((self.level - 1) % 5))

            # Se scendono troppo, resettali appena sopra i blocchi per dare una chance minima
            if self.enemyPosition >= 350:
                self.enemyPosition = 100

            # Rigenera nemici e blocker
            self.make_enemies()

            # Applica la nuova velocità
            self.enemies.moveTime = new_move_time

            # Aggiorna sprite group
            self.allSprites = pygame.sprite.Group(self.player, self.enemies, self.livesGroup, self.mysteryGroup, self.allBlockers)

            # Limite di sicurezza: se scendono troppo, resetta la posizione
            if self.enemyPosition >= BLOCKERS_POSITION:
                self.enemyPosition = ENEMY_DEFAULT_POSITION

            # Ricrea solo i nemici e i blocker, mantenendo Score e Vite
            self.make_enemies()
            self.make_blockers() # Opzionale: nel gioco originale i blocker NON si rigenerano

            # Aggiorna il gruppo di tutti gli sprite per includere i nuovi nemici
            self.allSprites = pygame.sprite.Group(self.player, self.enemies, self.livesGroup, self.mysteryGroup, self.allBlockers)

            # --- AGGIORNAMENTO SCORE GRAFICO ---
            # Qui uniamo il reward (che può essere negativo) al punteggio visivo
        self.score += reward

        # --- 4. DATA COLLECTION ---
        if self.collect_data and self.shipAlive:
             state_now = self.get_state()
             row = list(state_now) + [action]
             self.writer.writerow(row)

        # --- 5. RENDER (Opzionale) ---
        if WATCH_MODE:
            SCREEN.blit(self.background, (0, 0))
            self.allSprites.draw(SCREEN)
            self.bullets.draw(SCREEN)
            self.enemyBullets.draw(SCREEN)
            self.allBlockers.draw(SCREEN)
            self.explosionsGroup.draw(SCREEN)
            # HUD
            score_text = Text(FONT, 20, f"Score: {int(self.score)} | Lvl: {self.level}", GREEN, 5, 5)
            score_text.draw(SCREEN)
            pygame.display.update()
            self.clock.tick(60)

        # Gestione chiusura finestra
        for e in pygame.event.get():
            if e.type == QUIT:
                sys.exit()

        return self.get_state(), reward, done

# --- TEST DI FUNZIONAMENTO (Nessun Agente Reale) ---
if _name_ == '_main_':
    # Questo blocco serve solo a testare se l'ambiente non crasha
    env = SpaceInvadersEnvironment(collect_data=False)

    print("Avvio Test Ambiente (Random Actions)... Premi Ctrl+C per fermare.")
    state = env.reset()

    while True:
        # Genera azione casuale (0,1,2,3)
        action = randint(0, 3)

        # Esegui step
        next_state, reward, done = env.step(action)

        if reward != 0:
            print(f"Reward: {reward}")

        if done:
            print("Game Over! Resetting...")
            env.reset()