package com.sih.document_screening.model;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import jakarta.persistence.ElementCollection;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Entity
@Table(name = "verification_scores")
@Getter
@Setter
@NoArgsConstructor
public class VerificationScore {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @OneToOne
    @JoinColumn(name = "screening_id", nullable = false)
    private Screening screening;

    private Double faceMatchScore;         // 0.0 - 1.0
    private Double photoTamperingScore;    // 0.0 - 1.0 (ELA / Splice detection)
    private Double textManipulationScore;  // 0.0 - 1.0
    private Boolean metadataAnomalyFound;

    @ElementCollection(fetch = FetchType.EAGER)
    private List<String> flaggedReasons = new ArrayList<>();
}