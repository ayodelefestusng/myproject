# Django Core Imports
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model, login
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail, BadHeaderError
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes
from django.template.loader import render_to_string
from django.utils import timezone

# Models
from .models import Conversation, Message, LLM, Post, Insight

# NLP Processing
from .nlp_processor import process_message, process_message2  # Ensure this module exists

# Additional Utilities
import json
import pandas as pd
import csv
import openpyxl  # For Excel export
from django.core.paginator import Paginator

# Standard Library
import logging

# Uncomment if needed
# from .forms import *  
# from .models import Customer, Duration  

def home(request):
    return render(request, "home.html")


@login_required
def chat_home(request):
    # Get or create the active conversation for the user
    if request.user.is_authenticated:
        conversation, _ = Conversation.objects.get_or_create(
            user=request.user,
            is_active=True,
            defaults={'is_active': True}
        )
        # Deactivate any other active conversations for this user
        Conversation.objects.filter(
            user=request.user, is_active=True
        ).exclude(id=conversation.id).update(is_active=False)
    else:
        # For anonymous users, use session to track conversation
        session = request.session.get('conversation_id')
        conversation = Conversation.objects.filter(id=session).first() if session else None
        if not conversation:
            conversation = Conversation.objects.create(is_active=True)
            request.session['conversation_id'] = conversation.id

    messages = conversation.messages.all().order_by('timestamp')
    return render(request, 'chat.html', {'messages': messages})


from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
import json
import logging

@login_required
@csrf_exempt
def send_message(request):
    try:
        user_message = ''
        attachment = None

        # Handle JSON or form-data inputs
        if 'application/json' in request.content_type:
            try:
                if request.body:
                    data = json.loads(request.body)
                    user_message = data.get('message', '').strip()
            except json.JSONDecodeError:
                return JsonResponse({'status': 'error', 'response': 'Invalid JSON format'}, status=400)
        else:
            user_message = request.POST.get('message', '').strip()
            attachment = request.FILES.get('attachment')

        # Validate input
        if not user_message and not attachment:
            return JsonResponse({'status': 'error', 'response': 'Message or attachment is required'}, status=400)

        # Get or create conversation
        conversation, created = Conversation.objects.get_or_create(user=request.user, is_active=True, defaults={'is_active': True})

        # Deactivate other conversations if needed
        if not created:
            Conversation.objects.filter(user=request.user, is_active=True).exclude(id=conversation.id).update(is_active=False)

        # Save user message
        message = Message.objects.create(conversation=conversation, text=user_message, is_user=True)

        # Handle attachments
        file_content = None
        if attachment and hasattr(attachment, 'read'):
            message.attachment = attachment
            message.save()

            try:
                file_extension = attachment.name.split('.')[-1].lower()
                
                if file_extension in ['txt', 'csv', 'json', 'log']:
                    file_content = attachment.read().decode('utf-8')
                elif file_extension in ['jpg', 'png', 'jpeg']:
                    file_content = f"Uploaded an image: {attachment.name}"
                else:
                    file_content = f"Unsupported file type ({file_extension})."
            except Exception as e:
                file_content = f"Error reading attachment: {str(e)}"
                logging.error(f"Attachment processing error: {e}")

        # Select correct message processing function
        try:
            if attachment:
                latest_file = Message.objects.filter(attachment__isnull=False).order_by('-timestamp').first()
                file_path = latest_file.attachment.path if latest_file else None

                bot_response = process_message2(user_message, request.user, request.session.session_key, file_path) if file_path else "No uploaded file found."
            else:
                bot_response = process_message(user_message, request.user, request.session.session_key)

            if isinstance(bot_response, dict):
                bot_response = bot_response.get('messages', '')

            if not bot_response:
                bot_response = "I'm sorry, I couldn't process your request."
        
        except Exception as e:
            bot_response = f"Error processing message: {str(e)}"
            logging.error(f"Message processing failed: {e}")

        # Save bot response
        Message.objects.create(conversation=conversation, text=bot_response, is_user=False)

        return JsonResponse({'status': 'success', 'response': bot_response, 'attachment_url': message.attachment.url if attachment else None})

    except Exception as e:
        logging.error(f"Server error: {e}")
        return JsonResponse({'status': 'error', 'response': f"Server error: {str(e)}"}, status=500)


@login_required
@csrf_exempt
def send_message1234(request):
    try:
        # Initialize variables
        user_message = ''
        attachment = None

        # Handle JSON or form-data inputs
        if 'application/json' in request.content_type:
            try:
                if request.body:
                    data = json.loads(request.body)
                    user_message = data.get('message', '').strip()
            except json.JSONDecodeError:
                return JsonResponse(
                    {'status': 'error', 'response': 'Invalid JSON format'},
                    status=400
                )
        else:
            user_message = request.POST.get('message', '').strip()
            attachment = request.FILES.get('attachment')

        # Validate input
        if not user_message and not attachment:
            return JsonResponse(
                {'status': 'error', 'response': 'Message or attachment is required'},
                status=400
            )

        # Get or create conversation
        conversation, created = Conversation.objects.get_or_create(
            user=request.user,
            is_active=True,
            defaults={'is_active': True}
        )

        # Deactivate other conversations
        if not created:
            Conversation.objects.filter(
                user=request.user, 
                is_active=True
            ).exclude(id=conversation.id).update(is_active=False)

        # Save user message
        message = Message.objects.create(
            conversation=conversation,
            text=user_message,
            is_user=True
        )

        # Handle attachment processing
        if attachment:
            message.attachment = attachment
            message.save()

            # Use process_message2 when an attachment is present
            try:
                bot_response = process_message2(
                    user_message, 
                    request.user, 
                    request.session.session_key, 
                    attachment
                )
                if isinstance(bot_response, dict):
                    bot_response = bot_response.get('messages', '')
                if not bot_response:
                    bot_response = "I'm sorry, I couldn't process your request."
            except Exception as e:
                bot_response = f"Error processing message with attachment: {str(e)}"

        else:
            # Use process_message when no attachment is present
            try:
                bot_response = process_message(
                    user_message, 
                    request.user, 
                    request.session.session_key
                )
                if isinstance(bot_response, dict):
                    bot_response = bot_response.get('messages', '')
                if not bot_response:
                    bot_response = "I'm sorry, I couldn't process your request."
            except Exception as e:
                bot_response = f"Error processing message: {str(e)}"
        
        
        # if attachment and hasattr(attachment, 'read'):  # Ensure attachment is a file-like object
        #     message.attachment = attachment
        #     message.save()

        # # Use process_message2 when an attachment is present
        # try:
        #     bot_response = process_message2(
        #         user_message, 
        #         request.user, 
        #         request.session.session_key, 
        #         attachment
        #     )
        #     if isinstance(bot_response, dict):
        #         bot_response = bot_response.get('messages', '')

        #     if not bot_response:
        #         bot_response = "I'm sorry, I couldn't process your request."
        # except Exception as e:
        #     bot_response = f"Error processing message with attachment: {str(e)}"

        # else:  # Use process_message when no attachment is present
        #     try:
        #         bot_response = process_message(
        #             user_message, 
        #             request.user, 
        #             request.session.session_key
        #         )
        #         if isinstance(bot_response, dict):
        #             bot_response = bot_response.get('messages', '')

        #         if not bot_response:
        #             bot_response = "I'm sorry, I couldn't process your request."
        #     except Exception as e:
        #         bot_response = f"Error processing message: {str(e)}"
                
                
        
        
    
        # Save bot response
        Message.objects.create(
            conversation=conversation,
            text=bot_response,
            is_user=False
        )

        return JsonResponse({
            'status': 'success',
            'response': bot_response,
            'attachment_url': message.attachment.url if attachment else None
        })

    except Exception as e:
        return JsonResponse(
            {'status': 'error', 'response': f"Server error: {str(e)}"},
            status=500
        )




@login_required
@csrf_exempt
@csrf_exempt
@login_required
def send_messagev123a(request):
    try:
        # Initialize variables
        user_message = ''
        attachment = None

        # Handle JSON or form-data inputs
        if 'application/json' in request.content_type:
            try:
                if request.body:
                    data = json.loads(request.body)
                    user_message = data.get('message', '').strip()
            except json.JSONDecodeError:
                return JsonResponse(
                    {'status': 'error', 'response': 'Invalid JSON format'},
                    status=400
                )
        else:
            user_message = request.POST.get('message', '').strip()
            attachment = request.FILES.get('attachment')

        # Validate input
        if not user_message and not attachment:
            return JsonResponse(
                {'status': 'error', 'response': 'Message or attachment is required'},
                status=400
            )

        # Get or create conversation
        conversation, created = Conversation.objects.get_or_create(
            user=request.user,
            is_active=True,
            defaults={'is_active': True}
        )

        # Deactivate other conversations
        if not created:
            Conversation.objects.filter(
                user=request.user, 
                is_active=True
            ).exclude(id=conversation.id).update(is_active=False)

        # Save user message
        message = Message.objects.create(
            conversation=conversation,
            text=user_message,
            is_user=True
        )

        # Handle attachment processing  
        file_content = None
        if attachment and hasattr(attachment, 'read'):  # Ensure it's a file-like object
            message.attachment = attachment
            message.save()
            
            # Read the file content (text-based files)
            try:
                file_extension = attachment.name.split('.')[-1].lower()
                
                if file_extension in ['txt', 'csv', 'json', 'log']:
                    file_content = attachment.read().decode('utf-8')  # Read as text
                elif file_extension in ['jpg', 'png', 'jpeg']:
                    file_content = f"Uploaded an image: {attachment.name}"
                else:
                    file_content = f"File type ({file_extension}) not supported for reading."
            except Exception as e:
                file_content = f"Error reading attachment: {str(e)}"

            # Use process_message2 when an attachment is present
            # 

        else:  
            # Use process_message when no attachment is present
            try:
                bot_response = process_message(
                    user_message, 
                    request.user, 
                    request.session.session_key
                )
                # if isinstance(bot_response, dict):
                #     bot_response = bot_response.get('messages', '')
                bot_response = bot_response.get('messages', '') if isinstance(bot_response, dict) else bot_response
                if not bot_response:
                    bot_response = "I'm sorry, I couldn't process your request."
            except Exception as e:
                bot_response = f"Error processing message: {str(e)}"



        try:
            # Get the latest uploaded file
            latest_file = Message.objects.filter(attachment__isnull=False).order_by('-timestamp').first()

            if latest_file:
                file_path = latest_file.attachment.path  # Correctly reference the saved file path

                bot_response = process_message2(
                    user_message, 
                    request.user, 
                    request.session.session_key, 
                    file_path  # Pass the correct file path
                )

                if isinstance(bot_response, dict):
                    bot_response = bot_response.get('messages', '')

                if not bot_response:
                    bot_response = "I'm sorry, I couldn't process your request."
            else:
                bot_response = "No uploaded file found."

        except Exception as e:
            bot_response = f"Error processing message with attachment: {str(e)}"

        # Save bot response
        Message.objects.create(
            conversation=conversation,
            text=bot_response,
            is_user=False
        )

        return JsonResponse({
            'status': 'success',
            'response': bot_response,
            'attachment_url': message.attachment.url if attachment else None
        })

    except Exception as e:
        return JsonResponse(
            {'status': 'error', 'response': f"Server error: {str(e)}"},
            status=500
        )





@login_required
@csrf_exempt
@require_POST
def send_messagev2(request):
    try:
        # Initialize variables
        user_message = ''
        attachment = None
        
        # Check content type
        if 'application/json' in request.content_type:
            # Handle JSON data
            try:
                if request.body:
                    data = json.loads(request.body)
                    user_message = data.get('message', '').strip()
            except json.JSONDecodeError:
                return JsonResponse(
                    {'status': 'error', 'response': 'Invalid JSON format'},
                    status=400
                )
        else:
            # Handle form-data (including file uploads)
            user_message = request.POST.get('message', '').strip()
            attachment = request.FILES.get('attachment')

        # Validate we have either message or attachment
        if not user_message and not attachment:
            return JsonResponse(
                {'status': 'error', 'response': 'Message or attachment is required'},
                status=400
            )

        # Get or create conversation
        conversation, created = Conversation.objects.get_or_create(
            user=request.user,
            is_active=True,
            defaults={'is_active': True}
        )
        
        # Deactivate other conversations
        if not created:
            Conversation.objects.filter(
                user=request.user, 
                is_active=True
            ).exclude(id=conversation.id).update(is_active=False)

        # Save user message
        message = Message.objects.create(
            conversation=conversation,
            text=user_message,
            is_user=True
        )

        # Handle file attachment
        # if attachment:
        #     message.attachment = attachment
        #     message.save()

        # # Process message
        # try:
        #     bot_response = process_message(
        #         user_message, 
        #         request.user, 
        #         request.session.session_key
        #     )
        #     if isinstance(bot_response, dict):
        #         bot_response = bot_response.get('messages', '')
            
        #     if not bot_response:
        #         bot_response = "I'm sorry, I couldn't process your request."
        # except Exception as e:
        #     bot_response = f"Error processing message: {str(e)}"

        
        
        # Handle file attachment
        if attachment:
            message.attachment = attachment
            message.save()

    # Process message with attachment
        try:
            bot_response = process_message2(
                user_message, 
                request.user, 
                request.session.session_key, 
                attachment
            )
            if isinstance(bot_response, dict):
                bot_response = bot_response.get('messages', '')

            if not bot_response:
                bot_response = "I'm sorry, I couldn't process your request."
        except Exception as e:
            bot_response = f"Error processing message: {str(e)}"
        else:
            # Process message without attachment
            try:
                bot_response = process_message(
                    user_message, 
                    request.user, 
                    request.session.session_key
                )
                if isinstance(bot_response, dict):
                    bot_response = bot_response.get('messages', '')

                if not bot_response:
                    bot_response = "I'm sorry, I couldn't process your request."
            except Exception as e:
                bot_response = f"Error processing message: {str(e)}"

        # Save bot response
        Message.objects.create(
            conversation=conversation,
            text=bot_response,
            is_user=False
        )

        return JsonResponse({
            'status': 'success',
            'response': bot_response,
            'attachment_url': message.attachment.url if attachment else None
        })

    except Exception as e:
        return JsonResponse(
            {'status': 'error', 'response': f"Server error: {str(e)}"},
            status=500
        )




@login_required
@csrf_exempt
def send_messagev1(request):
    data = json.loads(request.body)
    user_message = data.get('message', '').strip()
    session = request.session.session_key
    if not session:
        request.session.create()
        session = request.session.session_key

    if not user_message:
        return JsonResponse({'status': 'error', 'response': 'Message cannot be empty'})

    # Get or create the active conversation
    conversation, _ = Conversation.objects.get_or_create(
        user=request.user,
        is_active=True,
        defaults={'is_active': True}
    )

    # Save user message
    Message.objects.create(
        conversation=conversation,
        text=user_message,
        is_user=True
    )

    # Process message and get bot response
    bot_response = process_message(user_message,request.user,session)
    bot_response = bot_response.get('messages', '') if isinstance(bot_response, dict) else bot_response
    print (f"Bot response: {bot_response}")
    if not bot_response:
        bot_response = "I'm sorry, I couldn't process your request."

    # Save bot response
    Message.objects.create(
        conversation=conversation,
        text=bot_response,
        is_user=False
    )

    return JsonResponse({'status': 'success', 'response': bot_response})





@login_required
@csrf_exempt
def send_message1(request,param_name):
    # data = json.loads(request.body)
    # user_message = data.get('message', '').strip()
    
    user_message =param_name
    if not user_message:
        return JsonResponse({'status': 'error', 'response': 'Message cannot be empty'})


    # Process message and get bot response
    bot_response = process_message(user_message)
    # Fetch metadata if available
    bot_response = bot_response.get('metadata', {}) if isinstance(bot_response, dict) else {}
    # bot_response = bot_response.get('messages', '') if isinstance(bot_response, dict) else bot_response
    print (f"Bot response: {bot_response}")
    if not bot_response:
        bot_response = "I'm sorry, I couldn't process your request."


    return JsonResponse({'status': 'success', 'response': bot_response})






@login_required
def chat_history(request):
    conversations = Conversation.objects.filter(user=request.user).order_by('-updated_at')
    return render(request, 'history.html', {'conversations': conversations})


def view_conversation(request, conversation_id):
    conversation = get_object_or_404(Conversation, id=conversation_id, user=request.user)
    messages = conversation.messages.all().order_by('timestamp')
    return render(request, 'chatbot/conversation.html', {
        'messages': messages,
        'conversation': conversation
    })


def edit_currency(request):
    category_id = request.POST.get('currency')
    cont = LLM.objects.get(pk=1)
    cont.currency = category_id
    cont.save()
    return redirect('chat.html')


@login_required
def oya(request):
    # insights = Insight.objects.filter(user=request.user).order_by('-id')[:1]
    insights = Insight.objects.all()
    return render(request, 'oya.html', {'insight': insights})

def oya1(request):
    insights = Insight.objects.all()
    return render(request, 'oya.html', {'insight': insights})


@login_required
def dashboard(request):
    return render(request, 'dashboard.html')


def post_list(request):
    posts = Post.objects.all()
    return render(request, 'post_list.html', {'posts': posts})

