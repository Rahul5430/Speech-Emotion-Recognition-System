import React, { useEffect, useState, useRef } from 'react';
import AudioPlayer from '../audioPlayer';
import uploadImg from '../../assets/cloud-upload-regular-240.png';
import './DragDropFile.css';

const DragDropFile = ({ predictFile, audioFile, setAudioFile }) => {
	const wrapperRef = useRef(null);
	const [fileDataURL, setFileDataURL] = useState(null);

	const onDragEnter = () => wrapperRef.current.classList.add('dragover');

	const onDragLeave = () => wrapperRef.current.classList.remove('dragover');

	const onDrop = () => wrapperRef.current.classList.remove('dragover');

	const onFileDrop = (e) => {
		e.preventDefault();
		const newFile = e.target.files[0];
		if (newFile) {
			setAudioFile(newFile);
		}
	};

	const fileRemove = (file) => {
		setAudioFile(null);
		setFileDataURL(null);
	};

	useEffect(() => {
		// let fileReader,
		// 	isCancel = false;
		// if (audioFile) {
		// 	fileReader = new FileReader();
		// 	fileReader.onload = (e) => {
		// 		const { result } = e.target;
		// 		if (result && !isCancel) {
		// 			setFileDataURL(result);
		// 		}
		// 	};
		// 	fileReader.readAsDataURL(audioFile);
		// 	console.log(URL.createObjectURL(audioFile));
		// }
		// return () => {
		// 	isCancel = true;
		// 	if (fileReader && fileReader.readyState === 1) {
		// 		fileReader.abort();
		// 	}
		// };
		if (audioFile) {
			setFileDataURL(URL.createObjectURL(audioFile));
		}
	}, [audioFile]);

	return (
		<React.Fragment>
			<form
				id='form-file-upload'
				encType='multipart/form-data'
				onSubmit={predictFile}
				ref={wrapperRef}
				className='drop-file-input'
				onDragEnter={onDragEnter}
				onDragLeave={onDragLeave}
				onDrop={onDrop}
			>
				<div className='drop-file-input__label'>
					<img src={uploadImg} alt='Upload' />
					<p>Drag & Drop your audio files here</p>
				</div>
				<input
					type='file'
					name='inputFile'
					id='inputFile'
					onChange={onFileDrop}
				/>
			</form>
			{audioFile && fileDataURL && (
				<div className='drop-file-preview'>
					<p className='drop-file-preview__title'>Ready to predict</p>
					<AudioPlayer
						audioSrc={fileDataURL}
						audioFile={audioFile}
						fileRemove={fileRemove}
					/>
				</div>
			)}
			{/* {audioFile && fileDataURL && (
				<audio src={URL.createObjectURL(audioFile)} controls />
			)} */}
			{audioFile && (
				<div className='text-center my-[20px]'>
					<button
						type='submit'
						form='form-file-upload'
						className='bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded'
					>
						Predict
					</button>
				</div>
			)}
		</React.Fragment>
	);
};

export default DragDropFile;
