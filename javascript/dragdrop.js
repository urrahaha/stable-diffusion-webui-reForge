// allows drag-dropping files into gradio image elements, and also pasting images from clipboard

function isValidImageList(files) {
    return files && files?.length === 1 && ['image/png', 'image/gif', 'image/jpeg'].includes(files[0].type);
}

let dragSourceElement = null;
let dragStartSelection = null;
let isDraggingWithShift = false;

window.document.addEventListener('dragstart', e => {
    if (dragDropTargetIsPrompt(e.target)) {
        dragSourceElement = e.target;
        dragStartSelection = {
            start: e.target.selectionStart,
            end: e.target.selectionEnd
        };
        // Check if Left Shift is held during drag start
        isDraggingWithShift = e.shiftKey && e.location === 1;
    }
});

function handleTextDragDrop(e) {
    const target = e.target;
    if (dragDropTargetIsPrompt(target)) {
        e.preventDefault();
        const text = e.dataTransfer.getData('text/plain');
        if (text && dragSourceElement) {
            const cursorPos = target.selectionStart;
            const textBefore = target.value.substring(0, cursorPos);
            const textAfter = target.value.substring(cursorPos);

            // Insert the dragged text at the cursor position
            target.value = textBefore + text + textAfter;
            
            // Move the cursor to the end of the inserted text
            target.selectionStart = target.selectionEnd = cursorPos + text.length;
            
            // Trigger an input event to ensure any listeners are notified
            target.dispatchEvent(new Event('input', { bubbles: true }));

            // If not dragging with Shift, remove the original text from the source
            if (!isDraggingWithShift) {
                const sourceStart = dragStartSelection.start;
                const sourceEnd = dragStartSelection.end;
                const sourceValue = dragSourceElement.value;
                const newSourceValue = sourceValue.slice(0, sourceStart) + sourceValue.slice(sourceEnd);
                dragSourceElement.value = newSourceValue;
                dragSourceElement.dispatchEvent(new Event('input', { bubbles: true }));
                
                // Adjust cursor position in source if dropping after the drag start point in the same field
                if (dragSourceElement === target && cursorPos > sourceStart) {
                    const shift = sourceStart - sourceEnd + text.length;
                    target.selectionStart += shift;
                    target.selectionEnd += shift;
                }
            }
        }
    }
    
    // Reset drag source and shift state after drop
    dragSourceElement = null;
    dragStartSelection = null;
    isDraggingWithShift = false;
}

window.document.addEventListener('drop', async e => {
    const target = e.composedPath()[0];
    const text = e.dataTransfer.getData('text/plain');

    if (dragDropTargetIsPrompt(target) && text) {
        handleTextDragDrop(e);
        return;
    }

function dropReplaceImage(imgWrap, files) {
    if (!isValidImageList(files)) {
        return;
    }

    const tmpFile = files[0];

    imgWrap.querySelector('.modify-upload button + button, .touch-none + div button + button')?.click();
    const callback = () => {
        const fileInput = imgWrap.querySelector('input[type="file"]');
        if (fileInput) {
            if (files.length === 0) {
                files = new DataTransfer();
                files.items.add(tmpFile);
                fileInput.files = files.files;
            } else {
                fileInput.files = files;
            }
            fileInput.dispatchEvent(new Event('change'));
        }
    };

    if (imgWrap.closest('#pnginfo_image')) {
        // special treatment for PNG Info tab, wait for fetch request to finish
        const oldFetch = window.fetch;
        window.fetch = async(input, options) => {
            const response = await oldFetch(input, options);
            if ('api/predict/' === input) {
                const content = await response.text();
                window.fetch = oldFetch;
                window.requestAnimationFrame(() => callback());
                return new Response(content, {
                    status: response.status,
                    statusText: response.statusText,
                    headers: response.headers
                });
            }
            return response;
        };
    } else {
        window.requestAnimationFrame(() => callback());
    }
}

function eventHasFiles(e) {
    if (!e.dataTransfer || !e.dataTransfer.files) return false;
    if (e.dataTransfer.files.length > 0) return true;
    if (e.dataTransfer.items.length > 0 && e.dataTransfer.items[0].kind == "file") return true;

    return false;
}

function dragDropTargetIsPrompt(target) {
    if (target?.placeholder && target?.placeholder.indexOf("Prompt") >= 0) return true;
    if (target?.parentNode?.parentNode?.className?.indexOf("prompt") > 0) return true;
    return false;
}

window.document.addEventListener('dragover', e => {
    const target = e.composedPath()[0];
    if (!eventHasFiles(e) && !e.dataTransfer.types.includes('text/plain')) return;

    var targetImage = target.closest('[data-testid="image"]');
    if (!dragDropTargetIsPrompt(target) && !targetImage) return;

    e.stopPropagation();
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
});

let dragSourceElement = null;
let dragStartSelection = null;

window.document.addEventListener('dragstart', e => {
    if (dragDropTargetIsPrompt(e.target)) {
        dragSourceElement = e.target;
        dragStartSelection = {
            start: e.target.selectionStart,
            end: e.target.selectionEnd
        };
    }
});

window.document.addEventListener('drop', async e => {
    const target = e.composedPath()[0];
    const url = e.dataTransfer.getData('text/uri-list');
    const text = e.dataTransfer.getData('text/plain');
    if (dragDropTargetIsPrompt(target) && text) {
        handleTextDragDrop(e);
        
        // Reset drag source after drop
        dragSourceElement = null;
        dragStartSelection = null;
        return;
    }
    
    if (!eventHasFiles(e) && !url) return;

    if (dragDropTargetIsPrompt(target)) {
        e.stopPropagation();
        e.preventDefault();

        const isImg2img = get_tab_index('tabs') == 1;
        let prompt_image_target = isImg2img ? "img2img_prompt_image" : "txt2img_prompt_image";

        const imgParent = gradioApp().getElementById(prompt_image_target);
        const files = e.dataTransfer.files;
        const fileInput = imgParent.querySelector('input[type="file"]');
        if (eventHasFiles(e) && fileInput) {
            fileInput.files = files;
            fileInput.dispatchEvent(new Event('change'));
        } else if (url) {
            try {
                const request = await fetch(url);
                if (!request.ok) {
                    console.error('Error fetching URL:', url, request.status);
                    return;
                }
                const data = new DataTransfer();
                data.items.add(new File([await request.blob()], 'image.png'));
                fileInput.files = data.files;
                fileInput.dispatchEvent(new Event('change'));
            } catch (error) {
                console.error('Error fetching URL:', url, error);
                return;
            }
        }
    }

    var targetImage = target.closest('[data-testid="image"]');
    if (targetImage) {
        e.stopPropagation();
        e.preventDefault();
        const files = e.dataTransfer.files;
        dropReplaceImage(targetImage, files);
        return;
    }
});

window.addEventListener('paste', e => {
    const files = e.clipboardData.files;
    if (!isValidImageList(files)) {
        return;
    }

    const visibleImageFields = [...gradioApp().querySelectorAll('[data-testid="image"]')]
        .filter(el => uiElementIsVisible(el))
        .sort((a, b) => uiElementInSight(b) - uiElementInSight(a));


    if (!visibleImageFields.length) {
        return;
    }

    const firstFreeImageField = visibleImageFields
        .filter(el => !el.querySelector('img'))?.[0];

    dropReplaceImage(
        firstFreeImageField ?
            firstFreeImageField :
            visibleImageFields[visibleImageFields.length - 1]
        , files
    );
})});
