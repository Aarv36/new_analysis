from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.contrib import messages
from .forms import UploadFileForm
from .models import UploadedFile
from .utils import analyze_pdf, extract_text_from_pdf, extract_text_from_word, find_details_in_text
import os
import shutil
import uuid
from django.http import HttpResponse, FileResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import zipfile
import logging
import cv2

# Configure logging
logger = logging.getLogger(__name__)

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                uploaded_file = request.FILES['file']
                uploaded_file_instance = UploadedFile(file=uploaded_file)
                uploaded_file_instance.save()

                file_path = uploaded_file_instance.file.path
                logger.info(f"Uploaded file path: {file_path}")

                if file_path.endswith('.pdf'):
                    extracted_text = extract_text_from_pdf(file_path)
                elif file_path.endswith('.docx'):
                    extracted_text = extract_text_from_word(file_path)
                elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
                    # Directly analyze the image for QR and Aadhar details
                    pass
                else:
                    messages.error(request, "Unsupported file type")
                    return redirect('upload_file')

                extracted_details = find_details_in_text(extracted_text)
                request.session['extracted_details'] = extracted_details

                return redirect('analyze_file', file_id=uploaded_file_instance.id)
            except Exception as e:
                logger.error(f"Error in upload_file: {e}")
                messages.error(request, f"Error: {e}")
        else:
            messages.error(request, "Invalid form submission")
    else:
        form = UploadFileForm()
    return render(request, 'analysis/upload.html', {'form': form})


def download_highlighted_images(request, file_id, format):
    uploaded_file = get_object_or_404(UploadedFile, id=file_id)
    output_folder = os.path.join(settings.MEDIA_ROOT, 'temp', str(file_id))
    highlighted_images = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.png')]

    if format == 'pdf':
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="highlighted_images_{file_id}.pdf"'

        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)

        for img_path in highlighted_images:
            p.drawImage(img_path, 0, 0, width=letter[0], height=letter[1])
            p.showPage()

        p.save()
        pdf = buffer.getvalue()
        buffer.close()
        response.write(pdf)
        return response

    elif format == 'zip':
        response = HttpResponse(content_type='application/zip')
        response['Content-Disposition'] = f'attachment; filename="highlighted_images_{file_id}.zip"'

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zip_file:
            for img_path in highlighted_images:
                zip_file.write(img_path, os.path.basename(img_path))
        buffer.seek(0)
        response.write(buffer.read())
        return response

    return HttpResponse(status=400)


def analyze_file(request, file_id):
    try:
        uploaded_file = get_object_or_404(UploadedFile, id=file_id)
        file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.file.name)
        logger.info(f"Analyzing file path: {file_path}")

        output_folder = os.path.join(settings.MEDIA_ROOT, 'temp', str(file_id))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        message, highlighted_pages, aadhar_details, face_details, page_statuses = analyze_pdf(file_path, output_folder)
        highlighted_images_urls = []

        for image_path in highlighted_pages:
            base_name = os.path.basename(image_path)
            unique_name = f"{uuid.uuid4().hex}_{base_name}"
            destination = os.path.join(output_folder, unique_name)
            shutil.move(image_path, destination)
            highlighted_images_urls.append(os.path.join(settings.MEDIA_URL, 'temp', str(file_id), unique_name))


        extracted_details = request.session.pop('extracted_details', {})

        # Combine details by page
        combined_details = {}
        for page_num, details in aadhar_details:
            if page_num not in combined_details:
                combined_details[page_num] = {'aadhar': [], 'faces': [], 'status': {}}
            combined_details[page_num]['aadhar'].extend(details)

        for page_num, faces in face_details:
            if page_num not in combined_details:
                combined_details[page_num] = {'aadhar': [], 'faces': [], 'status': {}}
            combined_details[page_num]['faces'].extend(faces)

        # Add statuses to combined details
        for page_num, status in page_statuses.items():
            if page_num not in combined_details:
                combined_details[page_num] = {'aadhar': [], 'faces': [], 'status': {}}
            combined_details[page_num]['status'] = status


        return render(request, 'analysis/result.html', {
            'message': message,
            'highlighted_images': highlighted_images_urls,
            'combined_details': combined_details,
            #'aadhar_details': aadhar_details,
            'extracted_details': extracted_details,
            #'face_details': face_details,
            'file_id': file_id
        })
    except Exception as e:
        logger.error(f"Error in analyze_file: {e}")
        messages.error(request, f"Error: {e}")
        return redirect('upload_file')

