from django.core.management.base import BaseCommand
from ai_models.services import get_chatbot_ai

class Command(BaseCommand):
    help = 'Build FAISS index from knowledge base'
    
    def handle(self, *args, **options):
        self.stdout.write('Building FAISS index...')
        try:
            chatbot = get_chatbot_ai()
            chatbot.build_faiss_index()
            self.stdout.write(
                self.style.SUCCESS('Successfully built FAISS index')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error building FAISS index: {str(e)}')
            )