document.addEventListener('DOMContentLoaded', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: 
            { facingMode:"user",width: { ideal: 640 },height: { ideal: 480 },frameRate : {ideal:30 , max:120}}
        });
        const recordedVideo = document.getElementById('recordedVideo');
        recordedVideo.setAttribute("playsinline",true);
        recordedVideo.srcObject = stream;
        recordedVideo.style.display = 'block';
        const urlParams = new URLSearchParams(window.location.search);
        const name = urlParams.get("name")
        const nric = urlParams.get("nric")

        
        const mediaRecorder = new MediaRecorder(stream);
        const chunks = [];
        mediaRecorder.ondataavailable = (event) => chunks.push(event.data);
        mediaRecorder.onstop = async () => {
            const blob = new Blob(chunks, { type: 'video/mp4' });
            const formData = new FormData();
            formData.append('video', blob, 'video_output.mp4');
            formData.append("name",name);
            formData.append("nric",nric)

            try {
                // Sending the video data to the backend
                const response = await fetch('/upload_video', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                console.log('Server response:', data);
                frames = [];
                console.log(data.name)
                console.log(data.nric)
                console.log(data.encodings_match)
                // Construct the new URL with the query parameter value
                if (data.encodings_match == true){
                    const URL_success = "http://localhost:5000/success?name=" + data.name + '&nric='+data.nric + '&match='+data.encodings_match;
                    // const URL_success = "https://kyc.mkad.my/success?name=" + data.name + '&nric='+data.nric + '&match='+data.encodings_match;
                    window.location.href = URL_success
                }
                else{
                    const URL_failed = "http://localhost:5000/failed?name=" + data.name + '&nric='+data.nric + '&match='+data.encodings_match;
                    // const URL_failed = "https://kyc.mkad.my/failed?name=" + data.name + '&nric='+data.nric + '&match='+data.encodings_match; 
                    window.location.href = URL_failed
                }

            });
            } catch (error) {
                console.error('Error uploading video:', error);
            }
        };

        // Auto-stop recording after 1 seconds
        setTimeout(() => {
            mediaRecorder.stop();
            // recordedVideo.style.display = 'none';
            // stream.getTracks().forEach(track => track.stop());
        }, 1000);

        mediaRecorder.start();
    } catch (error) {
        console.error('Error accessing camera and microphone:', error);
    }
});