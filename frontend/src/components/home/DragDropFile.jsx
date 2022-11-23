import React, { useRef } from 'react';
import uploadImg from '../../assets/cloud-upload-regular-240.png';
import musicImg from '../../assets/music-file.png';
import './DragDropFile.css';

const DragDropFile = ({ predictFile, audioFile, setAudioFile }) => {
	const wrapperRef = useRef(null);

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
	};

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
			{audioFile && (
				<div className='drop-file-preview'>
					<p className='drop-file-preview__title'>Ready to predict</p>
					<div className='drop-file-preview__item'>
						<img src={musicImg} alt='Audio File' />
						<div className='drop-file-preview__item__info'>
							<p>{audioFile.name}</p>
							<p>{audioFile.size}B</p>
						</div>
						<span
							className='drop-file-preview__item__del'
							onClick={() => fileRemove(audioFile)}
						>
							x
						</span>
					</div>
				</div>
			)}
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
