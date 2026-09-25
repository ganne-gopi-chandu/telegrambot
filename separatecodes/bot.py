# bot.py
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters
try:
    from rag import answer_question
except ImportError:
    from separatecodes.rag import answer_question
from dotenv import load_dotenv

load_dotenv()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("📩 Received /start command")
    await update.message.reply_text("Ask me about convex optimization 📘")

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.message.text
    print(f"📩 Received user query: '{q}'")
    msg = await update.message.reply_text("Thinking… 🤔")
    try:
        ans = answer_question(q)
        print("✅ Responded to user successfully.")
        await msg.edit_text(ans)
    except Exception as e:
        print(f"❌ Error processing question '{q}': {e}")
        await msg.edit_text(f"❌ Error: {e}")

bot_token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
if not bot_token:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is missing in .env")

print("\n🚀 Telegram Bot is live and listening for messages!")
print("👉 Open Telegram, search your bot, and send /start or any question.\n")

app = ApplicationBuilder().token(bot_token).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

app.run_polling()

